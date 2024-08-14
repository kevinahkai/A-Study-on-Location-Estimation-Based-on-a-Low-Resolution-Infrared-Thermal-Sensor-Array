import cv2
import numpy as np
import threading
import busio
import board
import time
import atexit
import signal
import sys
import adafruit_amg88xx
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib.animation import FuncAnimation
from skimage.morphology import erosion, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

# I2C initialization
i2c_bus = busio.I2C(board.SCL, board.SDA)

# Sensor initialization
sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x68)

# Wait for sensor initialization
time.sleep(0.1)

background_model = None  # Background model initialization
alpha = 0.02  # Background model update factor

# Create a zero array with the same shape as the background model
foreground_mask = np.zeros_like(background_model, dtype=bool)

# Global thermal threshold for determining human presence
global_threshold = 23

# Set subplots and adjust spacing
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.1)

# Camera initialization settings
cap = cv2.VideoCapture(0)  # 0 is usually the first camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 32)  # Set camera resolution(width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 32) # Set camera resolution(height)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate
camera_frame = None  # Save current image frame from camera
lock = threading.Lock() 
running = True

if not cap.isOpened():
    print("Failed to open the camera")
    exit(1)

def capture_camera():
    global camera_frame
    while running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with lock:
                camera_frame = frame

capture_thread = threading.Thread(target=capture_camera, daemon=True)
capture_thread.start()

def update_background(current_data, background, alpha):
    global foreground_mask

    if background is None:
        return current_data
    else:
        # Update background model, ignoring pixels marked as foreground
        background[~foreground_mask] = (1 - alpha) * background[~foreground_mask] + alpha * current_data[~foreground_mask]
    
        # Reset foreground mask
        foreground_mask.fill(False)
    
        return background

def segment_foreground(current_data, background):
    foreground = current_data - background
    foreground[foreground < 0] = 0
    return foreground

def adaptive_threshold(foreground, sensitivity_base=2.5, num_people=1):
    mean = np.mean(foreground)
    std = np.std(foreground)
    sensitivity_adjusted = sensitivity_base / np.sqrt(num_people + 1)  # Adding 1 to avoid division by zero
    threshold = mean + sensitivity_adjusted * std
    return foreground > threshold

def detect_blobs(foreground, threshold=0.5):
    global foreground_mask
    binary_img = (foreground > threshold).astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(binary_img)

    # Update foreground mask based on detected blobs
    if np.all(foreground_mask == False):
        foreground_mask = np.zeros_like(binary_img, dtype=bool)
    foreground_mask[labels_im > 0] = True
    return num_labels - 1, labels_im  # Subtract 1 to exclude the background label

def should_apply_watershed(contours):
    min_distance = 20
    for i, contour1 in enumerate(contours):
        for j, contour2 in enumerate(contours):
            if i >= j:
                continue
            x1, y1, w1, h1 = cv2.boundingRect(contour1)
            x2, y2, w2, h2 = cv2.boundingRect(contour2)
            center1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
            center2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
            distance = np.linalg.norm(center1 - center2)
            if distance < min_distance:
                return True
    return False

def apply_watershed(binary_img):
    binary_img = binary_img.astype(np.uint8)
    if np.count_nonzero(binary_img) == 0: 
        return np.zeros(binary_img.shape, dtype=np.int32)
    
    distance = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    if np.max(distance) == 0:
        return np.zeros(binary_img.shape, dtype=np.int32)
    
    local_max = peak_local_max(distance, min_distance=2, labels=binary_img)
    if local_max.size == 0:
        return np.zeros(binary_img.shape, dtype=np.int32)
    
    markers = cv2.connectedComponents(local_max.astype(np.uint8))[1]
    if markers.shape != binary_img.shape:
        markers = cv2.resize(markers, (binary_img.shape[1], binary_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    labels = watershed(-distance, markers, mask=binary_img)
    return labels

def heatmap(frame):
    global background_model
    try:
        sensordata = np.array(sensor.pixels)
    except Exception as e:
        print(f"Failed to read sensor data: {e}")
        return

    zoom_factor = 8
    bicubic_data = zoom(sensordata, zoom_factor, order=3)
    background_model = update_background(bicubic_data, background_model, alpha)

    # If no one is present (thermal value below threshold), display "No human detected"
    if np.max(bicubic_data) < global_threshold:
        axes[0].clear()
        axes[0].imshow(bicubic_data, cmap="inferno", interpolation="bicubic")
        axes[0].set_title("Thermal Data", fontsize=10)
        axes[0].set_xlim(0, bicubic_data.shape[1] - 1)
        axes[0].set_ylim(bicubic_data.shape[0] - 1, 0)
        axes[0].text(0.5, 0.5, "No human detected", horizontalalignment='center', verticalalignment='center',
                     transform=axes[0].transAxes, color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))
    else:
        foreground = segment_foreground(bicubic_data, background_model)
        binary_image = adaptive_threshold(foreground)
        num_blobs, labels_im = detect_blobs(binary_image)
        binary_image = adaptive_threshold(foreground, num_people=num_blobs)
        selem = disk(2)
        binary_image = erosion(binary_image, selem)
        
        contours, _ = cv2.findContours((labels_im > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if should_apply_watershed(contours):
            labels = apply_watershed(binary_image)
            num_blobs = np.max(labels)
        else:
            labels = labels_im

        axes[0].clear()
        axes[0].imshow(foreground, cmap="inferno", interpolation="bicubic")
        axes[0].set_title("Foreground Data", fontsize=10)
        axes[0].set_xlim(0, bicubic_data.shape[1] - 1)
        axes[0].set_ylim(bicubic_data.shape[0] - 1, 0)

        coordinates_text = ""
        for idx, contour in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            coordinates_text += f"{idx}. ({center_x}, {center_y})\n"
            rect = plt.Rectangle((x, y), w, h, edgecolor='white', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].text(center_x, center_y, str(idx), color='red', fontsize=10, ha='center', va='center')

        axes[0].text(0.05, 0.95, coordinates_text, horizontalalignment='left', verticalalignment='top',
                     transform=axes[0].transAxes, color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

        axes[1].clear()
        axes[1].imshow(binary_image, cmap="inferno", interpolation="bicubic")
        axes[1].set_title(f"Binary Image - {num_blobs} humans", fontsize=10)
        axes[1].set_xlim(0, binary_image.shape[1] - 1)
        axes[1].set_ylim(binary_image.shape[0] - 1, 0)

        for idx, contour in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            rect = plt.Rectangle((x, y), w, h, edgecolor='white', facecolor='none')
            axes[1].add_patch(rect)
            axes[1].text(center_x, center_y, str(idx), color='red', fontsize=10, ha='center', va='center')

        axes[1].text(0.05, 0.95, coordinates_text, horizontalalignment='left', verticalalignment='top',
                     transform=axes[1].transAxes, color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

    with lock:
        if camera_frame is not None:
            axes[2].clear()
            axes[2].imshow(camera_frame)
            axes[2].axis('off')
            axes[2].set_title("Camera View", fontsize=10)

ani = FuncAnimation(fig, heatmap, interval=100, cache_frame_data=False)
plt.show()

def cleanup():
    global running
    print("Cleaning up resources...")
    running = False
    capture_thread.join()
    cap.release()
    print("Camera released.")

atexit.register(cleanup)

def signal_handler(sig, frame):
    print(f"Signal {sig} received. Exiting gracefully.")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)  # Capture Ctrl+C signal
signal.signal(signal.SIGTERM, signal_handler)  # Capture terminal signal

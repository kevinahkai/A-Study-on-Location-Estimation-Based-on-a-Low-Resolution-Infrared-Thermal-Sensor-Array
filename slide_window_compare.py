import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import busio
import board
import adafruit_amg88xx
import time
from scipy.ndimage import gaussian_filter

# I2Cバスの初期化
i2c_bus = busio.I2C(board.SCL, board.SDA)

# センサーの初期化
sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x68)

# センサーの初期化待ち
time.sleep(0.1)

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
first_slide_window = []
second_slide_window = []
first_window_size = 3
second_window_size = 3

# 初始化熱力圖和 colorbar
sensordata = np.array(sensor.pixels)
imgs = []

for i in axes:
    img = i.imshow(sensordata, cmap="inferno", interpolation="bicubic")
    imgs.append(img)

# 添加水平colorbar至圖表下方
cbar = fig.colorbar(imgs[0], ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.04, pad=0.2)
cbar.set_label('Temperature')

# 調整子圖間距
plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.3, wspace=0.2, hspace=0.4)


def heatmap(frame):
    global first_slide_window, second_slide_window
    # データ取得
    sensordata = np.array(sensor.pixels)

    # Update first level queue
    first_slide_window.append(sensordata)
    if len(first_slide_window) > first_window_size:
        first_slide_window.pop(0)

    # Calculate first level moving average if enough data is collected
    if len(first_slide_window) == first_window_size:
        first_ma = np.mean(first_slide_window, axis=0)
        # Update second level queue
        second_slide_window.append(first_ma)
        if len(second_slide_window) > second_window_size:
            second_slide_window.pop(0)

    # 當前數據中最熱點的座標(人體位置)
    y, x = np.unravel_index(np.argmax(sensordata), sensordata.shape)

    # 更新左側的圖表
    axes[0].clear()
    axes[0].imshow(sensordata, cmap="inferno", interpolation="bicubic")
    axes[0].scatter(x, y, color='white', s=120)  # 標記最熱點
    axes[0].set_title("Original Data")
    axes[0].text(0.95, 0.95, f'Hotspot\nx: {x}\ny: {y}', 
                horizontalalignment='right', verticalalignment='top', 
                transform=axes[0].transAxes, color='white', fontsize=10)
    
    # Calculate moving average if enough data is collected
    if len(first_slide_window) == first_window_size:
        first_ma = np.mean(first_slide_window, axis=0)
        first_ma_y, first_ma_x = np.unravel_index(np.argmax(first_ma), first_ma.shape)
        axes[1].clear()
        axes[1].imshow(first_ma, cmap="inferno", interpolation="bicubic")
        axes[1].scatter(first_ma_x, first_ma_y, color='white', s=120)  # 標記最熱點
        axes[1].set_title("One Slide Moving Average")
        axes[1].text(0.95, 0.95, f'Hotspot\nx: {first_ma_x}\ny: {first_ma_y}', 
                     horizontalalignment='right', verticalalignment='top', 
                     transform=axes[1].transAxes, color='white', fontsize=10)
    else:
        axes[1].clear()
        axes[1].imshow(sensordata, cmap="inferno", interpolation="bicubic")
        axes[1].set_title("Original Data")

    # Calculate moving average if enough data is collected
    if len(second_slide_window) == second_window_size:
        second_ma = np.mean(second_slide_window, axis=0)
        second_ma_y, second_ma_x = np.unravel_index(np.argmax(second_ma), second_ma.shape)
        axes[2].clear()
        axes[2].imshow(second_ma, cmap="inferno", interpolation="bicubic")
        axes[2].scatter(second_ma_x, second_ma_y, color='white', s=120)  # 標記最熱點
        axes[2].set_title("Two Slide Moving Average")
        axes[2].text(0.95, 0.95, f'Hotspot\nx: {second_ma_x}\ny: {second_ma_y}', 
                     horizontalalignment='right', verticalalignment='top', 
                     transform=axes[2].transAxes, color='white', fontsize=10)
    else:
        axes[2].clear()
        axes[2].imshow(sensordata, cmap="inferno", interpolation="bicubic")
        axes[2].set_title("Original Data")

# 使用 FuncAnimation 持續更新圖表
ani = FuncAnimation(fig, heatmap, interval=100, cache_frame_data=False)  # 每100毫秒更新一次

plt.show()
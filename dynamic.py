import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import busio
import board
import adafruit_amg88xx
import time

# I2Cバスの初期化
i2c_bus = busio.I2C(board.SCL, board.SDA)

# センサーの初期化
sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x68)

# センサーの初期化待ち
time.sleep(0.1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
slide_window = []
window_size = 5

# 初始化熱力圖和 colorbar
sensordata = np.array(sensor.pixels)
imgs = []
for i in range(2):
    img = axes[i].imshow(sensordata, cmap="inferno", interpolation="bicubic")
    imgs.append(img)
    plt.colorbar(img, ax=axes[i])

def heatmap(frame):
    # データ取得
    sensordata = np.array(sensor.pixels)

    # Update queue
    slide_window.append(sensordata)
    if len(slide_window) > window_size:
        slide_window.pop(0)

    # 當前數據中最熱點的座標(人體位置)
    y, x = np.unravel_index(np.argmax(sensordata), sensordata.shape)

    # 更新左側的圖表
    axes[0].clear()
    axes[0].imshow(sensordata, cmap="inferno", interpolation="bicubic")
    axes[0].scatter(x, y, color='white', s=120)  # 標記最熱點
    axes[0].set_title("Original Data")

    # Calculate moving average if enough data is collected
    if len(slide_window) == window_size:
        ma = np.mean(slide_window, axis=0)
        ma_y, ma_x = np.unravel_index(np.argmax(ma), ma.shape)
        axes[1].clear()
        axes[1].imshow(ma, cmap="inferno", interpolation="bicubic")
        axes[1].scatter(ma_x, ma_y, color='white', s=120)  # 標記最熱點
        axes[1].set_title("Moving Average")
    else:
        axes[1].clear()
        axes[1].imshow(sensordata, cmap="inferno", interpolation="bicubic")
        axes[1].set_title("Original Data")

# 使用 FuncAnimation 持續更新圖表
ani = FuncAnimation(fig, heatmap, interval=100)  # 每100毫秒更新一次

plt.show()
import time
import busio
import board
import adafruit_amg88xx
import matplotlib.pyplot as plt
import numpy as np

# I2Cバスの初期化
i2c_bus = busio.I2C(board.SCL, board.SDA)

# センサーの初期化
sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x68)

# センサーの初期化待ち
time.sleep(.1)

# 8x8ピクセルの画像とbicubic補間をした画像を並べて表示させる
plt.subplots(figsize=(8, 4))

slide_window = []
window_size = 5

# 设置更新间隔 ##新
update_interval = 0.1  # 每0.1秒更新一次，以符合感测器的最大频率
last_time = time.time()

# ループ開始
while True:
    current_time = time.time()
    if current_time - last_time >= update_interval:
        # データ取得
        sensordata = np.array(sensor.pixels)

        # Update queue
        slide_window.append(sensordata)
        if len(slide_window) > window_size:
            slide_window.pop(0)

        # 當前數據中最熱點的座標(人體位置)
        y, x = np.unravel_index(np.argmax(sensordata), sensordata.shape)

        # 显示原始数据
        plt.subplot(1, 2, 1)
        plt.imshow(sensordata, cmap="inferno", interpolation="bicubic")
        plt.colorbar()
        plt.scatter(x, y, color='white', s=120)  # 標記最熱點
        plt.title("Original Data")
        
        # Calculate moving average if enough data is collected
        if len(slide_window) == window_size:
            ma = np.mean(slide_window, axis=0)
            ma_y, ma_x = np.unravel_index(np.argmax(ma), ma.shape)
            plt.subplot(1, 2, 2)
            plt.imshow(ma, cmap="inferno", interpolation="bicubic")
            plt.colorbar()
            plt.scatter(ma_x, ma_y, color='white', s=120)  # 標記最熱點
            plt.title("Moving Average")
        else:
            # 显示原始数据
            plt.subplot(1, 2, 2)
            plt.imshow(sensordata, cmap="inferno", interpolation="bicubic")
            plt.colorbar()
            plt.title("Original Data")

        # 更新显示
        plt.pause(0.1)
        # 清除之前的图像
        plt.clf()
        
        last_time = current_time
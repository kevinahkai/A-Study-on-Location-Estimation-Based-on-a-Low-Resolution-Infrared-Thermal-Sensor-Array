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

# ループ開始
while True:
    # データ取得
    #sensordata = sensor.pixels
    sensordata = np.array(sensor.pixels)

    #Continue update queue
    #這邊的視窗大小要重新設定，要慢慢實驗
    slide_window.append(sensordata)
    if len(slide_window) > 30:
        slide_window.pop(0)

    #Calculate moving average
    ma = np.mean(slide_window, axis=0) if len(slide_window) == 3 else None

    # 當前數據中最熱點的座標(人體位置)
    y, x = np.unravel_index(np.argmax(sensordata), sensordata.shape)

    # 移動平均值中的最熱點座標(人體位置)
    ma_y, ma_x = np.unravel_index(np.argmax(ma), ma.shape)

    # 8x8ピクセルのデータ
    plt.subplot(1, 2, 1)
    fig = plt.imshow(sensordata, cmap="inferno")
    plt.colorbar()
    plt.scatter(x, y, color='black', s=120)  # 標記最熱點
    plt.title("Original Data")
    # 顯示人體位置座標
    #plt.text(x, y, f'({x}, {y})', color='black', ha='right', va='bottom')
    plt.text(0.5, 0.5, f'Hotspot\nx: {x}\ny: {y}', horizontalalignment='center', verticalalignment='center')

    #Location Detection
    plt.subplot(1, 2, 2)
    fig = plt.imshow(ma, cmap="inferno")
    plt.colorbar()
    plt.scatter(ma_x, ma_y, color='black', s=120)  # 在移动平均图中标记最热点
    plt.title("Location Estimation")
    # 顯示人體位置座標
    plt.text(0.5, 0.5, f'Hotspot\nx: {ma_x}\ny: {ma_y}', horizontalalignment='center', verticalalignment='center')

    # 如果使用plt.show，會停止，所以使用pause
    # 如果不做plt.clf，會顯示很多彩條
    plt.pause(.1)
    plt.clf()
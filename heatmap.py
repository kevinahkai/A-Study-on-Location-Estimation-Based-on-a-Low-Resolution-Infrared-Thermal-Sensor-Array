import time
import busio
import board

import adafruit_amg88xx

import matplotlib.pyplot as plt

# I2Cバスの初期化
i2c_bus = busio.I2C(board.SCL, board.SDA)

# センサーの初期化
sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x68)

# センサーの初期化待ち
time.sleep(.1)

# 8x8ピクセルの画像とbicubic補間をした画像を並べて表示させる
plt.subplots(figsize=(8, 4))

# データ取得
sensordata = sensor.pixels

# 8x8ピクセルのデータ
plt.subplot(1, 2, 1)
fig = plt.imshow(sensordata, cmap="inferno")
plt.colorbar()

# bicubic補間したデータ
plt.subplot(1, 2, 2)
fig = plt.imshow(sensordata, cmap="inferno", interpolation="bicubic")
plt.colorbar()

plt.show()










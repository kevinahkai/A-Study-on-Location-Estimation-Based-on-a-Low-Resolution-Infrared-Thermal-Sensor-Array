import time

import busio
import board

import adafruit_amg88xx

# I2Cバスの初期化
i2c_bus = busio.I2C(board.SCL, board.SDA)

# センサーの初期化
# アドレスを68に指定
sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x68)

# センサーの初期化待ち
time.sleep(.1)

# 8x8の表示
print(sensor.pixels)

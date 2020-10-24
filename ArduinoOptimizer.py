
import serial
import os
from serial.tools import list_ports  # For listing available serial ports
import ctypes


user = ctypes.windll.user32

print('Detected serial ports:')
for d in serial.tools.list_ports.comports():
    print("Port:" + str(d.device) + "\tSerial Number:" + str(d.serial_number))

print('\n포트 번호를 입력주세요')
port = input()


ser = serial.Serial(port="COM{}".format(port), baudrate=9600)
#ser = serial.Serial(port="COM10", baudrate=9600)

os.system(os.path.dirname(os.path.realpath(__file__)) + "\\First_Brightness.bat")
cb = 100

def change_brightness(curr_bn, to_bn):
    if curr_bn > to_bn:
        while curr_bn > to_bn:
            os.system(os.path.dirname(os.path.realpath(__file__)) + "\\Decrease_Brightness.bat")
            curr_bn -= 10
    else:
        while curr_bn < to_bn:
            os.system(os.path.dirname(os.path.realpath(__file__)) + "\\Increase_Brightness.bat")
            curr_bn += 10
    return curr_bn


while True:
    if ser.readable():
        res = ser.readline()

        data = res.decode()[:-1]
        print(data)

        idx = data.index(':')
        data_type = data[:idx]
        data_value = data[idx + 1:]

        if data_type == 'IR':
            data_value = int(data_value)
            print(data_value)
            if data_value == 98:
                # Windows + L
                ctypes.windll.user32.LockWorkStation()
        elif data_type == 'BR':
            i_brightness = int(data_value)
            # i_brightness = int(data_value)

            print(i_brightness)
            if i_brightness <= 70:  # ~ 70 100
                cb = change_brightness(cb, 100)
            elif 70 < i_brightness <= 120:  # 70 ~ 120 90
                cb = change_brightness(cb, 90)
            elif 120 < i_brightness <= 170:
                cb = change_brightness(cb, 80)
            elif 170 < i_brightness <= 220:
                cb = change_brightness(cb, 70)
            elif 220 < i_brightness <= 270:
                cb = change_brightness(cb, 60)
            elif 270 < i_brightness <= 320:
                cb = change_brightness(cb, 50)
            elif 320 < i_brightness <= 370:
                cb = change_brightness(cb, 40)
            elif 370 < i_brightness <= 420:
                cb = change_brightness(cb, 30)
            elif 420 < i_brightness <= 470:
                cb = change_brightness(cb, 20)
            elif 470 < i_brightness <= 520:
                cb = change_brightness(cb, 10)
            elif 520 < i_brightness:
                cb = change_brightness(cb, 0)

        else:
            print('Err')


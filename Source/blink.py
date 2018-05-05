import RPi.GPIO as GPIO
import time

LedRed = 17
LedWhite = 19

def setup():
    GPIO.setmode(GPIO.BCM)

    GPIO.setup(LedRed, GPIO.OUT)
    #GPIO.output(LedRed, False)

    GPIO.setup(LedWhite, GPIO.OUT)
    #GPIO.output(LedWhite, False)

def red():
    GPIO.output(LedRed, True)
    time.sleep(1)
    GPIO.output(LedRed, False)
    time.sleep(1)

def white():
    GPIO.output(LedWhite, True)
    time.sleep(1)
    GPIO.output(LedWhite, False)
    time.sleep(1)

def destroy():
    GPIO.output(LedRed, False)
    GPIO.output(LedWhite, False)
    GPIO.cleanup()

def inputLaneDetection():
    try:
        while True:
            red()

    except KeyboardInterrupt:
        destroy()

# if __name__ == '__main__':
#     setup()
#     inputLaneDetection()
#     destroy()
        
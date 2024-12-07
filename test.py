from helper import alt_tab
import pyautogui
import time
import keyboard

alt_tab()
time.sleep(.2)
keyboard.press('h')
time.sleep(.024)
keyboard.release('h')
time.sleep(1)
keyboard.press('h')
time.sleep(.024)
keyboard.release('h')
time.sleep(.2)
alt_tab()
print('finiz')
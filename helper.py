from PIL import Image
import cv2
import numpy as np
import time
import pyautogui

def alt_tab():
    pyautogui.keyDown('alt')
    time.sleep(.2)
    pyautogui.press('tab')
    time.sleep(.2)
    pyautogui.keyUp('alt')
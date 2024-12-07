import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import ImageGrab, Image
import pyautogui
import win32gui
import cv2
import win32con
from torchvision import transforms
from collections import deque
import keyboard
import pyautogui

class PPLEnv(gym.Env):
    def __init__(self, state_bbox=(0, 0, 1920, 1080), mode='grey', stack_size=4):
        super(PPLEnv, self).__init__()
        self.action_space = spaces.Discrete(6)  # 6 possible actions: W, A, S, D, H, K

        # Bounding box for the state
        self.bbox = state_bbox
        self.cloud_bbox_dolphin = (390, 70, 845, 155)
        # self.window_handle = find_window(window_title="Pokemon Puzzle League (E) - Project64 3.0.1.5664-2df3434")

        # Determine the height and width based on the bounding box
        self.width = self.bbox[2] - self.bbox[0]
        self.height = self.bbox[3] - self.bbox[1]
        self.stack_size = stack_size  # Number of frames to stack

        # Initialize the frame stack
        self.frames = deque(maxlen=self.stack_size)

        # Screenshot History
        self.preprocessed_screenshot_history = []

        # Template Images
        self.template1_gray = cv2.cvtColor(cv2.imread("temp_img1.png"), cv2.COLOR_BGR2GRAY)
        self.template2_gray = cv2.cvtColor(cv2.imread("temp_img2.png"), cv2.COLOR_BGR2GRAY)
        # Select preprocessing function based on the mode
        if mode == 'grey':
            self.preprocess_frame = preprocess_frame_grey
            # Grayscale: Single channel
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
            )
        elif mode == 'color':
            self.preprocess_frame = preprocess_frame_color
            # Color: Three channels (RGB)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )
        else:
            raise ValueError("Invalid mode specified. Use 'grey' or 'color'.")
        

    def step(self, action):
        do_action(action=action)                    # do action
        observation = self.get_state() # new state
        done = self.is_game_over()            # check game over
        reward = self.calculate_reward()            # check reward
        return observation, reward, done, False, {} # return

    def reset(self, seed=None, options=None):
        # Reset the game (you need to implement this part)
        #self.reset_game()
        # Capture the initial screenshot as the initial observation (convert to grayscale)
        observation = self.get_state()
        return observation, {}

    def render(self):
        # Capture the initial screenshot as the initial observation (convert to grayscale)
        observation = self.get_state()
        return observation

    def calculate_reward(self):
        return 1  

    # should take 0.003s
    def is_game_over(self): 
        input_img = np.array(ImageGrab.grab(bbox=self.cloud_bbox_dolphin)) # Grab img and convert to numpy array
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY) # grayscale it
        # Perform template matching for both templates
        result1 = cv2.matchTemplate(input_gray, self.template1_gray, cv2.TM_CCOEFF_NORMED)
        result2 = cv2.matchTemplate(input_gray, self.template2_gray, cv2.TM_CCOEFF_NORMED)
        # Get the best match positions and scores for both templates
        _, max_val1, _, _ = cv2.minMaxLoc(result1)
        _, max_val2, _, _ = cv2.minMaxLoc(result2)
        # Take the maximum accuracy from both templates
        best_accuracy = max(max_val1, max_val2)
        # Accuracy Check
        if best_accuracy > 0.85:
            return True
        else:
            return False

    def reset_game(self):
        state = self.get_state()
        return state
    
    def get_state_old(self):
        #print(f"Window handle: {window_handle}") # sometimes you might need this line to fix weird win32gui.SetForegroundWindow(window_handle) bug
        win32gui.SetForegroundWindow(self.window_handle)
        # screenshot = ImageGrab.grab() # capture full screen
        screenshot = ImageGrab.grab(bbox=self.bbox)
        screenshot_np = np.array(screenshot)
        screenshot_np = self.preprocess_frame(screenshot_np)
        return screenshot_np
    
    def get_state(self):
        # Focus Window
        # win32gui.SetForegroundWindow(self.window_handle)
        # Grab Screenshot
        screenshot = ImageGrab.grab(bbox=self.bbox) # PIL
        # screenshot = pyautogui.screenshot(region=self.bbox) # pyautogui
        # screenshot = np.array(ImageGrab.grab(bbox=self.bbox)) # cv2
        # Convert to numpy array
        screenshot_np = np.array(screenshot)
        # Preprocess frame
        screenshot_np = self.preprocess_frame(screenshot_np)
        self.preprocessed_screenshot_history.append(screenshot_np)

        # Stack the frame
        self.frames.append(screenshot_np)
        
        # If we don't have enough frames yet, fill with zeros
        while len(self.frames) < self.stack_size:
            if screenshot_np.ndim == 2:  # Greyscale frame
                self.frames.append(np.zeros_like(screenshot_np))
            else:  # Color frame
                self.frames.append(np.zeros_like(screenshot_np))

        # Stack frames into a single tensor of shape (stack_size, height, width, channels)
        stacked_frames = np.stack(list(self.frames), axis=0)  # Shape: [stack_size, height, width, channels]
        
        # Adjust the channel dimension
        if screenshot_np.ndim == 2:  # Grayscale
            stacked_frames = np.expand_dims(stacked_frames, axis=-1)  # Add channel dimension: [stack_size, height, width, 1]
        return stacked_frames

# def find_window(window_title:str):
#     # Get the handle of the window
#     hwnd = win32gui.FindWindow(None, window_title)
#     if hwnd == 0:
#         raise Exception(f"Window with title '{window_title}' not found.")
#     return hwnd


def get_score_img():
    score_screenshot = ImageGrab.grab(bbox=(440, 50, 510, 160)) # capture the score number
    score_screenshot_np = np.array(score_screenshot)
    return score_screenshot_np


def do_action(action:int):
    # Simulate key press and release for the action
    if action == 0:
        press_and_release('w')  # Press and release 'w'
    elif action == 1:
        press_and_release('a')  # Press and release 'a'
    elif action == 2:
        press_and_release('s')  # Press and release 's'
    elif action == 3:
        press_and_release('d')  # Press and release 'd'
    elif action == 4:
        press_and_release('h')  # Press and release 'h'
    elif action == 5:
        press_and_release('k')  # Press and release 'k'


def press_and_release(key=None):
    keyboard.press(key)
    time.sleep(.025)
    keyboard.release(key)

# split up in color and not ig 
def check_score(image, template_path='0_temp.png'):
    # # Load template (image of "0")
    # template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # Read template as grayscale

    # # Convert the game screen to grayscale
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Match template
    # result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)

    # # Get the best match position and value
    # _, max_val, _, _ = cv2.minMaxLoc(result)

    # return max_val
    return 0

# Preprocess the frames (resize and convert to grayscale)
def preprocess_frame_color(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((64, 64)),  # Resize to 84x84
        transforms.ToTensor(),  # Convert to tensor, will have shape [1, 84, 84]
    ])
    return transform(frame).squeeze(0)  # Remove the single channel dimension, resulting in [84, 84]

# Preprocess the frames (resize and convert to grayscale)
def preprocess_frame_grey(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((256, 256)),  # Resize to 84x84
        transforms.ToTensor(),  # Convert to tensor, will have shape [1, 84, 84]
    ])
    return transform(frame).squeeze(0)  # Remove the single channel dimension, resulting in [84, 84]
    
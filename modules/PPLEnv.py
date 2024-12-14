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
import ctypes


from modules.read_memory import get_process_id, read_memory, PROCESS_ALL_ACCESS

class PPLEnv(gym.Env):
    def __init__(self, base_address, state_bbox=(0, 0, 1920, 1080), color_mode='grey', stack_size=4, cursor_logging=False, screenshot_saving=False, load_state_key='f2'):
        super(PPLEnv, self).__init__()
        # Action Space
        self.action_space = spaces.Discrete(5)  # 6 possible actions: W, A, S, D, H, K

        # Image Size
        self.img_size_color = (32, 32)
        
        # Logging
        self.cursor_logging = cursor_logging
        self.save_screenshots = screenshot_saving

        # Bounding boxes 
        self.state_bbox = state_bbox # bounding box for state
        self.bbox_bottom_left_cell_spotlight = (400, 965, 440, 1005) # bounding box for check game over

        # Determine the height and width based on the bounding box
        self.width = self.state_bbox[2] - self.state_bbox[0]
        self.height = self.state_bbox[3] - self.state_bbox[1]
        self.stack_size = stack_size  # Number of frames to stack

        # Initialize the frame stack
        self.frames = deque(maxlen=self.stack_size)

        # Keys
        self.load_state_key = load_state_key

        # Screenshot History
        self.screenshot_history = []
        self.preprocessed_screenshot_history = []

        # Template Images
        self.template_cell_gray = cv2.cvtColor(cv2.imread("images/temp_img_cell.png"), cv2.COLOR_BGR2GRAY)

        # Track episode score for calc reward
        self.episode_score = 0

        # Select preprocessing function based on the mode
        if color_mode == 'grey':
            self.preprocess_frame = preprocess_frame_grey
            # Grayscale: Single channel
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
            )
        elif color_mode == 'color':
            self.preprocess_frame = preprocess_frame_color
            # Color: Three channels (RGB)
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )
        else:
            raise ValueError("Invalid mode specified. Use 'grey' or 'color'.")
        
        # # Memory Reading Setup
        # process_name = "DolphinMemoryEngine.exe"
        # process_id = get_process_id(process_name)
        # self.process_handle = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, process_id)
        # # Score Setup
        # base_address_score = 0x7FFA08340000 # 7FFA08340000 - Qt6Gui.dll
        # base_offset_score = 0x006EB448
        # offsets_score = [0x80, 0xE30]

        # first_pointer = base_address_score + base_offset_score
        # dereferenced_address  = read_memory(self.process_handle, first_pointer, data_type=ctypes.c_uint64)
        # dereferenced_address_hex = (hex(dereferenced_address))

        # for offset in offsets_score:
        #     pointer = dereferenced_address + offset
        #     dereferenced_address  = read_memory(self.process_handle, pointer)
        #     dereferenced_address_hex = (hex(dereferenced_address))
        # self.score_pointer = pointer
        # # Cursor Horizontal Setup
        # base_address_ch = 0x7FFA08340000 # 7FFA08340000 - Qt6Gui.dll
        # base_offset_ch = 0x006EA080
        # offsets_ch = [0x110, 0xB0, 0X8, 0X1C0]

        # first_pointer = base_address_ch + base_offset_ch
        # dereferenced_address  = read_memory(self.process_handle, first_pointer, data_type=ctypes.c_uint64)
        # dereferenced_address_hex = (hex(dereferenced_address))

        # for offset in offsets_ch:
        #     pointer = dereferenced_address + offset
        #     dereferenced_address  = read_memory(self.process_handle, pointer)
        #     dereferenced_address_hex = (hex(dereferenced_address))
        # self.cursor_horizontal_pointer = pointer
        # # Cursor Vertical Setup
        # base_address_cv = 0x7FFA08340000 # 7FFA08340000 - Qt6Gui.dll
        # base_offset_cv = 0X006EA868
        # offsets_cv = [0x5F0, 0x30, 0X2E0]

        # first_pointer = base_address_cv + base_offset_cv
        # dereferenced_address  = read_memory(self.process_handle, first_pointer, data_type=ctypes.c_uint64)
        # dereferenced_address_hex = (hex(dereferenced_address))

        # for offset in offsets_cv:
        #     pointer = dereferenced_address + offset
        #     dereferenced_address  = read_memory(self.process_handle, pointer)
        #     dereferenced_address_hex = (hex(dereferenced_address))
        # self.cursor_vertical_pointer = pointer

        # # Test memory pointers
        # ch, cv = self.get_cursor_pos()
        # score = self.get_score()
        # print(f"Environment succesfully initialized \n" + f"- cursor: ({ch}, {cv})\n" + f"- score: {score}")
        
    # FUNCTIONS
    def step(self, action):
        do_action(action=action)                    # do action
        observation = self.get_state()              # new state
        done = self.is_game_over()                  # check game over
        reward = self.calculate_reward()            # check reward
        return observation, reward, done, False, {} # return

    def reset(self, seed=None, options=None):
        # Restart from save state
        self.preprocessed_screenshot_history = [] # resetting history to prevent extremely large arrays of images
        keyboard.press(self.load_state_key) # load save state in slot f2
        time.sleep(.025) # wait to release key
        keyboard.release(self.load_state_key)
        time.sleep(2.2) # wait for dolphin to update
        self.episode_score = 0 # reset total episode reward
        observation = self.get_state()
        return observation, {}

    def render(self):
        # Capture the initial screenshot as the initial observation (convert to grayscale)
        observation = self.get_state()
        return observation

    def calculate_reward(self):
        total_score = self.get_score()
        reward = total_score - self.episode_score
        self.episode_score+=reward
        return reward


    # should take 0.003s
    def is_game_over(self): 
        input_img = np.array(ImageGrab.grab(bbox=self.bbox_bottom_left_cell_spotlight)) # Grab img and convert to numpy array
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY) # grayscale it
        # Perform template matching for both templates
        result1 = cv2.matchTemplate(input_gray, self.template_cell_gray, cv2.TM_CCOEFF_NORMED)
        # Get the best match positions and scores for both templates
        _, max_val1, _, _ = cv2.minMaxLoc(result1)
        # Accuracy Check
        if max_val1 > 0.8:
            return False
        else:
            return True

    def reset_game(self):
        state = self.get_state()
        return state
    
    def get_state(self):
        # Grab Screenshot
        screenshot = ImageGrab.grab(bbox=self.state_bbox)  # PIL
        screenshot_np = np.array(screenshot)
        screenshot_np = self.preprocess_frame(screenshot_np, self.img_size_color)
        cursor_position_horizontal, cursor_position_vertical = self.get_cursor_pos()
        if self.save_screenshots:
            self.screenshot_history.append(screenshot)
            self.preprocessed_screenshot_history.append(screenshot_np)
        return [screenshot_np, cursor_position_horizontal, cursor_position_vertical]

    def get_cursor_pos(self):
        cursor_horizontal = read_memory(self.process_handle, self.cursor_horizontal_pointer, data_type=ctypes.c_uint8)
        cursor_vertical = read_memory(self.process_handle, self.cursor_vertical_pointer, data_type=ctypes.c_uint32)
        if self.cursor_logging:
            print(f'Cursor pos: ({cursor_horizontal}, {cursor_vertical})')
        return cursor_horizontal, cursor_vertical
    
    def get_score(self):
        return read_memory(self.process_handle, self.score_pointer, data_type=ctypes.c_uint32)


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
    # elif action == 5:
    #     press_and_release('k')  # Press and release 'k'


def press_and_release(key=None):
    keyboard.press(key)
    time.sleep(.025)
    keyboard.release(key)

# Preprocess the frames (resize and convert to grayscale)
def preprocess_frame_color(frame, img_size_color):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize(img_size_color),  # Resize to 84x84
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
    
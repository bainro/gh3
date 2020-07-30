import PIL
import cv2
import numpy as np
import pyautogui
import timeit

num_iters = 100
t = timeit.Timer('img = pyautogui.screenshot()', 'import pyautogui')
total_time = t.timeit(num_iters)
avg_time = total_time / num_iters
print("average time in seconds to take a screenshot using pyautogui: ", avg_time)
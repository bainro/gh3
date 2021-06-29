import cv2
import sys
import os
import numpy as np

# the region of screen with all 5 notes assuming 800x600 screen resolution
height = 29
width = 230
top = 369
left = 286
bottom = int(top) + int(height)
right = int(left) + int(width)

save_path = "./unlabeled_notes"
if not os.path.exists(save_path):
    os.makedirs(save_path)

count = 0
note_width = width//5 # 46px as of 8/2020
video_path = os.path.join(sys.argv[1])
# print(video_path)
video_cap = cv2.VideoCapture(video_path)
# set to one to start from the beginning
for i in range(300):
    success, image = video_cap.read()
#print(success)

single_note = np.zeros((46, 46, 3), dtype=np.float32)

while success:
    # trim down to region of interest(s)
    roi = image[top:bottom, left:right, :]
    for i in range(5):
        start_w = (i*note_width)
        stop_w = start_w + note_width
        single_note[0:height, 0:note_width, :] = roi[0:height, start_w:stop_w, :]
        filename = "frame_" + str(count) + "_" + str(i) + ".jpg"
        filename = os.path.join(save_path, filename)
        cv2.imwrite(filename, single_note)      
    success, image = video_cap.read()
    print("count: ", count)
    count += 1
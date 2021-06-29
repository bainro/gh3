import time
import cv2
import mss
import numpy as np

# the region of the screen with all 5 fret keys
reg = {
        "top": 480, 
        "left": 215, 
        "width": 370, 
        "height": 80
    }

def screen_record_efficient():
    fps = 0
    sct = mss.mss()
    last_time = time.time()

    while time.time() - last_time < 1:
        img = np.asarray(sct.grab(reg))
        fps += 1

        # cv2.imshow(title, img)
        # if cv2.waitKey(25) & 0xFF == ord("q"):
        #     cv2.destroyAllWindows()
        #     break

    cv2.imwrite("test.jpg", img)

    return fps, img

def speed_test_alt():
    sct = mss.mss()
    start_time = time.time()
    print("starting...")

    while time.time() - start_time < 10:
        start = time.perf_counter()
        img = np.asarray(sct.grab(reg))
        inference_time = time.perf_counter() - start
        print('%.1fms' % (inference_time * 1000))
        time.sleep(0.25)

def test_numpy_copy():
    # testing speeds of adding 6 columns to screen grab to make it square
    import random
    for _i in range(20):
        small = np.ones((74, 80), dtype=np.float32)
        big = np.zeros((80, 80), dtype=np.float32)
        start = time.perf_counter()
        big[0:74, 0:80] = small
        # essentially fractions of a nano second...
        print('%.1fms' % ((time.perf_counter() - start) * 1000))

if __name__ == '__main__':
    #_fps, img = screen_record_efficient()
    speed_test_alt()
    #test_numpy_copy()

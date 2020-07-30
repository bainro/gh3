import time
import cv2
import mss
import numpy


def screen_record_efficient():
    # 390x276 windowed mode
    # this is the size we need for GH3
    mon = {
        "top": 100, 
        "left": 100, 
        "width": 390, 
        "height": 276
        }

    title = "[MSS] FPS benchmark"
    fps = 0
    sct = mss.mss()
    last_time = time.time()

    while time.time() - last_time < 1:
        img = numpy.asarray(sct.grab(mon))
        fps += 1

        cv2.imshow(title, img)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    return fps

print("MSS:", screen_record_efficient())
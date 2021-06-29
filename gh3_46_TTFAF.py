r"""Example using TF Lite to classify a given single_note using an Edge TPU."""
import argparse
from PIL import Image
import numpy as np
import math
import cv2
import classify
import tflite_runtime.interpreter as tflite
import platform, mss, threading, queue, os, time
import direct_keyboard_inputs as k

# the region of screen with all 5 notes assuming 800x600 screen resolution
roi = {
  "top": 369, 
  "left": 286, 
  "width": 230, 
  "height": 29
}
roi_q = queue.Queue()
roi_v_q = queue.Queue()
note_q = queue.Queue()
note_width = 46
NOTES = [k.GREEN, k.RED, k.YELLOW, k.BLUE, k.ORANGE]

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def notes_worker():
  prev_notes, notes = note_q.get()
  for key in prev_notes:
    k.ReleaseKey(key)
  for key in notes:
    k.PressKey(key)
  k.PressKey(k.STRUM)
  time.sleep(0.025) # doesn't block other threads!
  k.ReleaseKey(k.STRUM)
  time.sleep(0.025)
  note_q.task_done()

def infer_worker(interpreter, threshold, video):
  single_note = np.zeros((46, 46, 3), dtype=np.float32)
  last_strum = count = 0
  final_count = math.inf
  last_infer_all_neg = True
  last_notes = []
  last_notes_non_empty = []
  note_delay = 0.26

  while count != final_count:
    roi_ = roi_q.get()
    # not an image but a expected frame count
    if type(roi_) is int:
      print("final frame count: " + str(roi_))
      final_count = roi_
      roi_q.task_done()
      continue
    count += 1 # must be after final frame count bit
    if video:
      roi_, timestamp = roi_
    current_notes = []
    start_i = -1 * note_width
    stop_i = 0
    wait_a_frame = False

    for i in range(5):
      start_i += note_width
      stop_i += note_width
      single_note[0:29, 0:note_width, :] = roi_[0:29, start_i:stop_i, :]
      #cv2.imwrite("test.jpg", single_note); break
      #t_test = time.perf_counter()
      classify.set_input(interpreter, single_note)
      interpreter.invoke()
      classes = classify.get_output(interpreter, 1, threshold)
      # t___ = time.perf_counter() - t_test
      # print(t___ * 1000)
      # if the highest probable class is "click" and over a threshold confidence:
      if len(classes) > 0 and classes[0][0] == 0:
        current_notes.append(NOTES[i])
        # if video:
        #   roi_ = cv2.putText(img=np.copy(roi_), text='%.2f' % classes[0][1], org=(16+46*(i), 20), fontScale=0.3,
        #                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255), thickness=1) 
        # skip the other notes since we most benefit from using the second detection
        if last_infer_all_neg:
          last_infer_all_neg = False
          wait_a_frame = True

    if video: 
      # really in the way, should add space @ the bottom for this info, if needed
      # roi_ = cv2.putText(img=np.copy(roi_), text='%.2f' % timestamp, org=(5,10), fontScale=0.3, 
      #                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0,0,255), thickness=1)
      roi_v_q.put(roi_)

    if wait_a_frame: continue

    can_strum_again = time.perf_counter() - last_strum > 0.04 # 0.05 
    
    tmp_last_notes = list(current_notes)
    current_notes += last_notes
    # remove potential dupes
    current_notes = list(dict.fromkeys(current_notes))
    last_notes = tmp_last_notes

    if len(current_notes):
      last_infer_all_neg = False 
      if can_strum_again:
        last_strum = time.perf_counter()
        if video:
          if final_count is math.inf: 
            note_q.put([last_notes_non_empty, current_notes])
            threading.Timer(interval=note_delay, function=notes_worker, args=[]).start()
        else:
          note_q.put([last_notes_non_empty, current_notes])
          threading.Timer(interval=note_delay, function=notes_worker, args=[]).start()
        last_notes_non_empty = list(current_notes)
    else:
      last_infer_all_neg = True

    roi_q.task_done()
  print("infer_worker finished at " + str(time.time()))
  
def video_worker(fps):
  codec = cv2.VideoWriter_fourcc(*'mp4v')
  video = cv2.VideoWriter("./test.mp4", codec, fps, (230, 29)) # 130
  print("video_worker initialized!")
  
  count = 0
  final_count = math.inf
  while count != final_count:
    img = roi_v_q.get()
    # how many frames to expect
    if type(img) is int:
      final_count = img
      roi_v_q.task_done()
      continue
    video.write(img)
    count += 1
    roi_v_q.task_done()

  video.release()
  print("video_worker finished at " + str(time.time()))

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  try:
    _interpreter = tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])
  except ValueError:
    print("must be linux...")
    _interpreter = tflite.Interpreter(
      model_path=model_file)
  return _interpreter

def release_keys():
  # clear all the still pressed keys
  k.ReleaseKey(k.STRUM)
  k.ReleaseKey(k.STAR)
  for n in NOTES:
    k.ReleaseKey(n)
    time.sleep(0.1)

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-d', '--duration', type=int, default=30, help='How long the agent plays')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0, help='Classification score threshold')
  parser.add_argument(
      '-v', '--roi_video', type=int, default=0, help='Record video @ specified FPS')
  args = parser.parse_args()

  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  def eval_folder(dir, class_toggle=True):
    assert os.path.exists(dir)

    wrong_path = "./wrong"
    if not os.path.exists(wrong_path):
      os.makedirs(wrong_path)

    below_thresh_path = "./below_thresh"
    if not os.path.exists(below_thresh_path):
      os.makedirs(below_thresh_path)

    gfr_files = os.listdir(dir)
    gh3_pics = []

    # keep only the jpg images
    for file in gfr_files:
        if file.endswith(".jpg") or file.endswith(".png"):
            gh3_pics.append(file)

    for path in gh3_pics:
      pic_path = os.path.join(dir, path)
      pic = cv2.imread(pic_path, cv2.COLOR_BGR2RGB)
      pic = np.array(pic, dtype=np.uint8)
      classify.set_input(interpreter, pic)
      interpreter.invoke()
      classes = classify.get_output(interpreter, 1, 0)
      if class_toggle: 
        i = 0 # click
      else:
        i = 1 # no click
      if classes[0][0] == i:
        if classes[0][1] < args.threshold:
          print("below threshold of " + str(args.threshold) + ": " + pic_path)
          filename = os.path.join(below_thresh_path, path)
          cv2.imwrite(filename, pic)
      else:
        print("wrong classification: " + pic_path)
        filename = os.path.join(wrong_path, path)
        cv2.imwrite(filename, pic)
  
  def live_play():
    print("SCRIPT STARTED")
    count_ = 0
    sct = mss.mss() # init screen grab object
    if args.roi_video:
      v = threading.Thread(target=video_worker, args=[args.roi_video])
      v.start()
      time.sleep(3)
    #threading.Thread(target=notes_worker, daemon=True).start()
    i = threading.Thread(target=infer_worker, args=(interpreter, args.threshold, args.roi_video), daemon=False)
    i.start()  

    start_time = time.time()
    while time.time() - start_time < args.duration:
      #t_test = time.perf_counter()
      #print(t_test)
      all_notes = np.asarray(sct.grab(roi))[:,:,:-1] # RGBA, so omit alpha
      # _ = time.perf_counter()
      # i_time = _ - t_test
      # print(i_time * 1000)
      if args.roi_video:
        roi_q.put((all_notes, time.perf_counter_ns()))
      else:
        roi_q.put(all_notes)
      count_ += 1     

    release_keys()
    roi_q.put(count_)
    if args.roi_video:
      roi_v_q.put(count_)
      #print("waiting for roi_video to finish")
      v.join()
    #print("waiting for inference worker to finish")
    i.join()
    print("Script finished! Avg FPS: " + str(count_/args.duration))

  #eval_folder("/home/rbain/links/fast_storage/python/gh3/data/click_and_no_click/noclick/", False)
  live_play()

if __name__ == '__main__':
  main()
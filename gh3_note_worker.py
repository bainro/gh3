r"""Example using TF Lite to classify a given single_note using an Edge TPU."""
import argparse
from PIL import Image
import numpy as np
import cv2
import classify
import tflite_runtime.interpreter as tflite
import platform, mss, threading, queue, os, time
import direct_keyboard_inputs as k

# the region of screen with all 5 notes assuming 800x600 screen resolution
roi = {
  "top": 480, 
  "left": 215, 
  "width": 370, 
  "height": 80
}
note_width = 74
NOTES = [k.GREEN, k.RED, k.YELLOW, k.BLUE, k.ORANGE]
note_q = queue.Queue()

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def notes_worker():
  prev_notes = []
  while True:
    notes = note_q.get()
    for key in prev_notes:
      k.ReleaseKey(key)
    for key in notes:
      k.PressKey(key)
    k.PressKey(k.STRUM)
    time.sleep(0.017)
    k.ReleaseKey(k.STRUM)
    prev_notes = notes

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
      '-d', '--duration', required=False, type=int, default=15.0, help='How long the agent plays')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
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
    count = 0
    single_note = np.zeros((80, 80, 3), dtype=np.float32)
    sct = mss.mss() # init screen grab object
    threading.Thread(target=notes_worker, daemon=True).start()
    start_time = last_strum = time.time()

    print("SCRIPT STARTED")

    while time.time() - start_time < args.duration:
      all_notes = np.asarray(sct.grab(roi))[:,:,:-1] # RGBA, so omit alpha
      # cv2.imwrite("test.jpg", all_notes); break
      current_notes = []
      start_i = -1 * note_width
      stop_i = 0
      count += 1

      for i in range(5):
        start_i += note_width
        stop_i += note_width
        single_note[0:80, 0:note_width, :] = all_notes[0:80, start_i:stop_i, :]
        classify.set_input(interpreter, single_note)
        interpreter.invoke()
        classes = classify.get_output(interpreter, 1, args.threshold)  
        # if the highest probable class is "click" and over a threshold confidence:
        if len(classes) > 0 and classes[0][0] == 0:
          current_notes.append(NOTES[i])

      can_strum_again = time.time() - last_strum > 0.1
      if len(current_notes) and can_strum_again:
        note_q.put(current_notes)

    print("SCRIPT END! FPS: " + str(count/args.duration))
    release_keys()

  #eval_folder("/home/rbain/links/fast_storage/python/gh3/data/click_and_no_click/noclick/", False)
  live_play()


if __name__ == '__main__':
  main()
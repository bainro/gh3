import pyautogui
from time import sleep

DELAY_BETWEEN_COMMANDS = 1.00

def main():
    initialize()
    sleep(4)

    pyautogui.press("a")
    pyautogui.press("enter")
    pyautogui.press("enter")
    pyautogui.press("enter")
    pyautogui.press("enter")                
    # pyautogui.keyDown("d")
    # pyautogui.keyDown(" ")

    # pyautogui.press('shiftright')
    # pyautogui.press('shiftright')
    # pyautogui.press('shiftright')

    # pyautogui.keyUp("a")
    # pyautogui.keyUp("d")
    # pyautogui.keyUp("esc")

    screen_size = pyautogui.size()
    print(screen_size)

def initialize():
    # https://pyautogui.readthedocs.io/en/latest/introduction.html
    # When fail-safe mode is True, moving the mouse to the upper-left corner will abort your program.
    pyautogui.FAILSAFE = True

def hold_key(key, seconds=1.00):
    pyautogui.keyDown(key)
    sleep(seconds)
    pyautogui.keyUp(key)

def get_mouse_pos(seconds=10):
    for i in range(0, seconds):
        print(pyautogui.position())
        sleep(1)


if __name__ == "__main__":
    main()
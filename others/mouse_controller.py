from pynput import keyboard
import pyautogui as pg
import os
import sys

MOUSE_DRAG_DISTANCE = 200
MOUSE_DRAG_DURATION = 0.3
MOUSE_SCROLL_DISTANCE = 10

screen_w, screen_h = pg.size()


def on_press(key):
    try:
        print('alphanumeric key "{0}" pressed'.format(key.char))
        if 'u' == key.char:
            pg.drag(0, +MOUSE_DRAG_DISTANCE, MOUSE_DRAG_DURATION, button='left')
        elif 'd' == key.char:
            pg.drag(0, -MOUSE_DRAG_DISTANCE, MOUSE_DRAG_DURATION, button='left')
        elif 'l' == key.char:
            pg.drag(+MOUSE_DRAG_DISTANCE, 0, MOUSE_DRAG_DURATION, button='left')
        elif 'r' == key.char:
            pg.drag(-MOUSE_DRAG_DISTANCE, 0, MOUSE_DRAG_DURATION, button='left')
        elif 'i' == key.char:
            pg.scroll(+MOUSE_SCROLL_DISTANCE)
        elif 'o' == key.char:
            pg.scroll(-MOUSE_SCROLL_DISTANCE)
        pg.moveTo(screen_w / 2, screen_h / 2, 0)

    except AttributeError:
        print('special key {0} pressed'.format(key))


def on_release(key):
    print('{0} released'.format(key))
    if key == keyboard.Key.esc:
        return False  # Stop listener


def main():
    sys.stdout = open(os.devnull, 'w')

    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
    ) as listener:
        listener.join()


if __name__ == '__main__':
    main()

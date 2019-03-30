import cv2

from constants import IMG_FILENAME

FRAME_RATE = 60
IMG_FOLDER = '../img/eyes_up_down'


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # get frame size
    size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(size)

    counter = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        # cv2.imshow('frame',gray)

        img_filename = IMG_FILENAME.format(
            IMG_FOLDER=IMG_FOLDER,
            INDEX=counter
        )
        counter += 1

        cv2.imwrite(filename=img_filename, img=frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if counter == 180:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

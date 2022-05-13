import cv2

from processing import process


def main():
    cap = cv2.VideoCapture(0)
    process(cap, camera=True)

if __name__ == '__main__':
    main()
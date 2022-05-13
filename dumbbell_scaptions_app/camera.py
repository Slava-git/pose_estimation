import cv2

from processing import process

def dumbbell_scaptions_counter():
    '''Count dumbbell scaptions and provide rules how do it properly'''


def main():
    cap = cv2.VideoCapture(0)
    process(cap, camera=True)


if __name__ == '__main__':
    main()
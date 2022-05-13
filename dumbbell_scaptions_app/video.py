import cv2
import argparse

from processing import process

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        help=('path to video'))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    path_to_video = args.path

    cap = cv2.VideoCapture(path_to_video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)

    output = cv2.VideoWriter('output/output_dumbell_scaptions.avi', 
                            cv2.VideoWriter_fourcc('M','J','P','G'), 24, frame_size)

    process(cap, video=True, output= output)

if __name__ == '__main__':
    main()
import cv2
import numpy as np
import sys
import argparse
import os

parent_path = os.path.dirname(os.getcwd())
sys.path.append(parent_path)

from settings import MP_DRAWING, MP_POSE
from calculating import calculate_angle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        help=('path to video'))
    args = parser.parse_args()
    return args


def catch_squats():
    '''Get images when angle is closest to 90 for each squat'''

    args = parse_args()
    path_to_video = args.path

    cap = cv2.VideoCapture(path_to_video)

    min_angle = 180
    counter = 0
    stage = None

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)

    output = cv2.VideoWriter('output/output_squats.avi',
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 23.98, frame_size)

    with MP_POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret == True:

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark

                    left_hip = [landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].y]

                    left_knee = [landmarks[MP_POSE.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[MP_POSE.PoseLandmark.LEFT_KNEE.value].y]

                    left_ankle = [landmarks[MP_POSE.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[MP_POSE.PoseLandmark.LEFT_ANKLE.value].y]

                    angle_left_knee = int(calculate_angle(
                        left_hip, left_knee, left_ankle))

                    right_hip = [landmarks[MP_POSE.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[MP_POSE.PoseLandmark.RIGHT_HIP.value].y]

                    right_knee = [landmarks[MP_POSE.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[MP_POSE.PoseLandmark.RIGHT_KNEE.value].y]

                    right_ankle = [landmarks[MP_POSE.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[MP_POSE.PoseLandmark.RIGHT_ANKLE.value].y]

                    angle_right_knee = int(calculate_angle(
                        right_hip, right_knee, right_ankle))

                    if (angle_left_knee > 160 or angle_right_knee > 160) and stage == "Squat":
                        stage = "Stand"
                        counter += 1
                        min_angle = 180

                    if angle_left_knee < 100 or angle_right_knee < 100:
                        stage = "Squat"

                except:
                    pass

                cv2.putText(image, f'Counter: {counter}', (15, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1, cv2.LINE_AA)

                cv2.putText(image, f'Stage: {stage}', (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 1, cv2.LINE_AA)

                # Visualize angles
                cv2.putText(image, str(angle_left_knee),
                            tuple(np.multiply(left_knee, [
                                frame_width, frame_height]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                0, 0, 0), 1, cv2.LINE_AA
                            )

                cv2.putText(image, str(angle_left_knee),
                            tuple(np.multiply(right_knee, [
                                    frame_width, frame_height+15]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                0, 0, 0), 1, cv2.LINE_AA
                            )


                MP_DRAWING.draw_landmarks(image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
                                          MP_DRAWING.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          MP_DRAWING.DrawingSpec(
                                              color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                if (stage == "Squat" and
                    (abs(angle_left_knee - 90) < min_angle) or
                        (abs(angle_right_knee - 90) < min_angle)):

                        min_angle = min(abs(angle_left_knee - 90),
                                        abs(angle_right_knee - 90))
                        cv2.imwrite(f'output/image_{counter}.jpg', image)
                output.write(image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            else:
                break

    cap.release()
    output.release()


if __name__ == '__main__':
    catch_squats()

import cv2
import sys
import numpy as np
import os

parent_path = os.path.dirname(os.getcwd())
sys.path.append(parent_path)

from calculating import calculate_angle
from settings import MP_DRAWING, MP_POSE


def process(cap, camera=False, video=False, output=None):
    '''Count dumbbell scaptions and provide rules how do it properly

    Params:
        cap(video capture object): captured video
        camera(bool): flag in order to use webcam
        video(bool): flag in order to use video
        output(video writer object): output video
    '''
    
    counter = 0
    stage = "Down"
    delay = 0
    move = "Hands Up"
    min_angle = 180

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    resolution = [640, 480] if camera==True else [frame_width, frame_height]

    with MP_POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret==True:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                previous_stage = stage

                try:
                    landmarks = results.pose_landmarks.landmark

                    left_shoulder = [landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y]


                    left_elbow = [landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].x,
                                    landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].y]

                    left_wrist = [landmarks[MP_POSE.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[MP_POSE.PoseLandmark.LEFT_WRIST.value].y]

                    angle_left_hand = int(calculate_angle(
                        left_shoulder, left_elbow, left_wrist))

                    left_hip = [landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].y]

                    angle_torso = calculate_angle(
                        left_elbow, left_shoulder, left_hip)

                    left_wrist = [landmarks[MP_POSE.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[MP_POSE.PoseLandmark.LEFT_WRIST.value].y]

                    right_shoulder = [landmarks[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[MP_POSE.PoseLandmark.RIGHT_SHOULDER.value].y]

                    right_elbow = [landmarks[MP_POSE.PoseLandmark.RIGHT_ELBOW.value].x,
                                    landmarks[MP_POSE.PoseLandmark.RIGHT_ELBOW.value].y]

                    right_wrist = [landmarks[MP_POSE.PoseLandmark.RIGHT_WRIST.value].x,
                                    landmarks[MP_POSE.PoseLandmark.RIGHT_WRIST.value].y]

                    angle_right_hand = int(calculate_angle(
                        right_shoulder, right_elbow, right_wrist))

                    if angle_torso <= 50.0:
                        if stage == "Up":
                            counter += 1
                            move = 'Hands Up'

                        stage = "Down"
                            
                        min_angle = 180.0

                    if angle_torso >= 85.0 and stage == "Down":
                        stage = "Up"
                        move = 'Hands Down'

                    if previous_stage != stage:
                        delay = 33

                    delay -= 1


                    if delay <= 0:
                        cv2.putText(image, f"{move}",
                                    tuple(np.multiply(left_wrist,[
                                            resolution[0], resolution[1]+70]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (
                                            0, 128, 0), 1, cv2.LINE_AA
                                    )

                    else:
                        cv2.putText(image, "Stay",
                                    tuple(np.multiply(left_wrist,[
                                            resolution[0], resolution[1]+70]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (
                                            0, 128, 0), 1, cv2.LINE_AA
                                    )

                    cv2.putText(image, f'Counter: {counter}', (15, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.putText(image, str(angle_left_hand),
                                tuple(np.multiply(left_elbow,
                                        resolution).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                        0, 0, 0), 1, cv2.LINE_AA
                    )

                    cv2.putText(image, str(angle_right_hand),
                                tuple(np.multiply(right_elbow,
                                        resolution).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                        0, 0, 0), 1, cv2.LINE_AA
                    )

                except:
                    pass


                MP_DRAWING.draw_landmarks(image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
                        MP_DRAWING.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2),
                        MP_DRAWING.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2))

                if video==True:
                    if stage == "Up" and (180.0 - angle_left_hand) < min_angle:
                        print(f'Check_{counter}')
                        min_angle = 180.0 - angle_left_hand
                        cv2.imwrite(f'output/image_{counter}.jpg', image)
                    
                    output.write(image)

                if camera==True:
                    cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            else:
                break

        cap.release()
        if video==True:
            output.release()
        if camera==True:
            cv2.destroyAllWindows()
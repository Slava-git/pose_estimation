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

def catch_dumbbell_scaptions():
    '''Get images when arms are extended'''

    args = parse_args()
    path_to_video = args.path

    cap = cv2.VideoCapture(path_to_video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width,frame_height)

    stage = None
    counter = 0
    min_angle = 180

    output = cv2.VideoWriter('output/output_dumbell_scaptions.avi', 
                            cv2.VideoWriter_fourcc('M','J','P','G'), 23.98, frame_size)

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
                    
                    left_shoulder = [landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[MP_POSE.PoseLandmark.LEFT_SHOULDER.value].y]

                    left_elbow = [landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[MP_POSE.PoseLandmark.LEFT_ELBOW.value].y]

                    left_wrist = [landmarks[MP_POSE.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[MP_POSE.PoseLandmark.LEFT_WRIST.value].y]

                    angle_hand = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    
                    left_hip = [landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[MP_POSE.PoseLandmark.LEFT_HIP.value].y]

                    angle_torso = calculate_angle(left_elbow, left_shoulder, left_hip)

                    if angle_torso <= 50.0:
                        if stage == "Up":
                            counter +=1

                        stage = "Down"
                        min_angle = 180.0

                    if angle_torso >= 85.0 and stage=="Down" :
                        stage = "Up"


                    cv2.putText(image, str(angle_hand), 
                                    tuple(np.multiply(left_elbow, [frame_width, frame_height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )

                    cv2.putText(image, str(angle_torso), 
                                    tuple(np.multiply(left_shoulder, [frame_width, frame_height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                        )   
                except:
                    pass
                
                cv2.putText(image, f'Stage: {stage}', (15,20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                cv2.putText(image, f'Counter: {counter}', (15,55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if stage == "Up" and (180.0 - angle_hand) < min_angle:
                    min_angle = 180.0 - angle_hand
                    cv2.imwrite(f'output/image_{counter}.jpg', image)
                    
                MP_DRAWING.draw_landmarks(image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
                                        MP_DRAWING.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        MP_DRAWING.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )
                                
                output.write(image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            else:
                break
        cap.release()
        output.release()

if __name__ == '__main__':
    catch_dumbbell_scaptions()
import cv2
import numpy as np
import sys        
import os

parent_path = os.path.dirname(os.getcwd())
sys.path.append(parent_path)

from calculating import calculate_angle
from settings import MP_DRAWING, MP_POSE


def squat_counter():
    '''Count squats and calculate angles beetween knee, hip, ankle real-time '''
    
    counter = 0
    stage = None

    cap = cv2.VideoCapture(0)

    with MP_POSE.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
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

                angle_left_knee = int(calculate_angle(left_hip, left_knee, left_ankle))
                
                right_hip = [landmarks[MP_POSE.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[MP_POSE.PoseLandmark.RIGHT_HIP.value].y]

                right_knee = [landmarks[MP_POSE.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[MP_POSE.PoseLandmark.RIGHT_KNEE.value].y]

                right_ankle = [landmarks[MP_POSE.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[MP_POSE.PoseLandmark.RIGHT_ANKLE.value].y]

                angle_right_knee = int(calculate_angle(right_hip, right_knee, right_ankle))

                if (angle_left_knee > 160.0 or angle_right_knee> 160.0) and stage=="Squat":
                    stage = "Stand"
                    counter +=1
                
                if angle_left_knee < 100.0 or angle_right_knee<100 :
                    stage = "Squat"
                
                cv2.putText(image, f'Counter: {counter}', (15,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(image, f'Stage: {stage}', (15,40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(image, str(angle_left_knee), 
                            tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                                    )

                cv2.putText(image, str(angle_left_knee), 
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                                    )

            except:
                pass

            MP_DRAWING.draw_landmarks(image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS,
                                    MP_DRAWING.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    MP_DRAWING.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    squat_counter()
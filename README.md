# Pose estimation for squats and dumbbell scaptions
This repo aims to pose estimate and count squats or dumbbell scaptions and gives some help how do it properly.

# Requirements
* Python>=3.7.0  
* `pip install -r requirements.txt`

# Dumbbell scaptions
## Webcam
`cd dumbbell_scaptions_app`    
`python camera.py`  
Process video directly via webcam realtime

## Video
`cd dumbbell_scaptions_app`    
`python video.py --path path/video.mp4`  
Process video you downloaded and return images when angle of arm is closest to 180 degree
### Parameters
* --path - path to video

# Squats
## Webcam
`cd squat_app`    
`python camera.py`  
Process video directly via webcam realtime

## Video
`cd squat_app`  
`python video.py --path path/video.mp4`  
Process video you downloaded and return images when angle beetween knee, hip, ankle is closest to 90 degree
### Parameters
* --path - path to video

# Results
Refer to `squat_app/output`, `dumbbell_scaptions_app/output` to see results.

import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Specify the path to your video file
video_path = 'video.mov'

# Initialize video capture
vidcap = cv2.VideoCapture(video_path)

# Set the desired window width and height
winwidth = 350
winheight = 600
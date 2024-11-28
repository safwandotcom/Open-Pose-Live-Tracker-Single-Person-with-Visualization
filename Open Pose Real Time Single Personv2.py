# -----------------------------------------------------------------------------
# Code created by: Mohammed Safwanul Islam @safwandotcom®
# Project: Computer Vision Data Science OPENPOSE 
# Date created: 16th November 2024
# Organization: N/A
# -----------------------------------------------------------------------------
# Description:
# This code captures live video from the webcam, applies pose estimation using MediaPipe, 
# and visualizes the detected body landmarks and connections in real time. The program runs continuously until the user presses the 'q' key to exit. 
# It demonstrates an application of computer vision for human pose tracking, which can be used in fields like fitness, gaming, and gesture recognition.
#   VERSION 2 visualizes the data into a white pop-up window which opens after the program is successfully running.
#### This code can detect only single person in detection ####
# -----------------------------------------------------------------------------
# License:
# This code belongs to @safwandotcom®.
# Code can be freely used for any purpose with proper attribution.
# -----------------------------------------------------------------------------
# Modules to install for this program to run using WINDOWS POWERSHELL
# pip install opencv-python
# pip install mediapipe


import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # Use 1 for external webcam, 0 for internal webcam

# Customize drawing styles for landmarks and connections
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=int(1.5))  # Green for landmarks
connection_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=int(2.5))  # Red for connections

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image from BGR (OpenCV default) to RGB (MediaPipe input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    pose_results = pose.process(image_rgb)

    # Create a black background to draw the pose landmarks and connections
    black_background = 255 * np.ones(image.shape, dtype=np.uint8)

    # If pose landmarks are detected, draw them on the black background
    if pose_results.pose_landmarks:
        # Draw the pose landmarks and connections on the black background
        mp_drawing.draw_landmarks(
            black_background, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,  # Pose connection lines
            landmark_style,            # Green for landmarks
            connection_style           # Red for connections
        )

    # Display the black background with pose landmarks and connections
    cv2.imshow('Open Pose Output by Safwanul', black_background)

    # Draw the landmarks and connections on the original image
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, 
            pose_results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,  # Pose connection lines
            landmark_style,            # Green for landmarks
            connection_style           # Red for connections
        )

    # Display the original webcam feed with landmarks and connections
    cv2.imshow('Webcam of Safwanul', image)

    # Exit on pressing the 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

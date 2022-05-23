import cv2
import mediapipe as mp
import time
import numpy as np

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpHands = mp.solutions.hands
hands = mpHands.Hands()

is_computer_macbook = 1

path_to_file = 'ping_pong.mp4'
cap = cv2.VideoCapture(path_to_file)

interesting_edges = {
    "LEFT_ARM": (11,13), #0
    "TRAPEZOID": (11,12), #1
    "RIGHT_ARM": (12,14), #3
    "LEFT_FOREARM": (13,15), #4
    "RIGHT_FOREARM": (14,16), #5
}
max_t = 500

# TODO - Change, do not make it realtime, computational resources might not be enough to support it

history = np.zeros(shape=(5, max_t, 3)) # The first index takes the same indexing as interesting_edges
t = 0

pTime = 0
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        pose_connections = mpPose.POSE_CONNECTIONS
        # TODO - Hard code it first and then change it
        for i in [11, 12, 13, 14, 15, 16]:
            base_idx = 11
            vector = np.array([
                results.pose_landmarks.landmark[i].x,
                results.pose_landmarks.landmark[i].y,
                results.pose_landmarks.landmark[i].x
            ])
            history[i - 11, t, :] = vector


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    # print(f'Current FPS is {fps}') # Currently with a value of 14, increase if you want to capture sport actions
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    t += 1
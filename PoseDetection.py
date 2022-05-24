import cv2
import mediapipe as mp
import time
import numpy as np
from tempfile import TemporaryFile


"""
PROCEDURE:
1. Take for every vector several observation of a ping pong serve
2. Normalize them by identifying the starting (x,y,z) vector
3. Subtract the increments in space from the following ones
4. Treat (x0, y1, z0) as a possible starting point (on which to sample an initial position from)
5. Treat the following trajectories [(x1,y1,z1), ..., (xn,yn,zn)] as a process of correlated observations"""


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpHands = mp.solutions.hands
hands = mpHands.Hands()

is_computer_macbook = 1

path_to_file = 'ping_pong_trimmed.mp4'
cap = cv2.VideoCapture(path_to_file)

interesting_edges = {
    "LEFT_ARM": (11,13), #0
    "TRAPEZOID": (11,12), #1
    "RIGHT_ARM": (12,14), #2
    "LEFT_FOREARM": (13,15), #3
    "RIGHT_FOREARM": (14,16), #4
}
max_t = 11 # TODO - Cange this automatically according to the number of frames in the image

# TODO - Change, do not make it realtime, computational resources might not be enough to support it

history = np.zeros(shape=(5, max_t, 3)) # The first index takes the same indexing as interesting_edges
t = 0

pTime = 0
success=True
while success:
    success, img = cap.read()
    if not success: break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # TODO - Hard code it first and then change it
        for i in [11, 12, 13, 14, 15, 15]:
            base_idx = 11
            vector = np.array([
                results.pose_landmarks.landmark[i].x,
                results.pose_landmarks.landmark[i].y,
                results.pose_landmarks.landmark[i].z
            ])
            history[i - 11, t, :] = vector

    cv2.putText(img, str(int(t)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.imwrite(f"output/Frame_{t}.png", img)
    cv2.waitKey(1)
    t += 1

# Now build the function which is going to be responsible for analyzing it

np.save(file="output", arr=history)
print(f'Number of frames is {t}')
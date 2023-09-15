import cv2
import mediapipe as mp
import numpy as np

frame_shape = (480, 640, 3) #specific to own camera
mask = np.zeros(frame_shape, dtype='uint8') #To draw permanently
colour = (120, 12, 20)
thickness = 4

prevxy = None
#model for tracking hand movements
hands = mp.solutions.hands
hand_landmark = hands.Hands(max_num_hands=1)

#video capturing process
cap = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils
while True:
    _,frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    op = hand_landmark.process(rgb)

    if op.multi_hand_landmarks:
        for all_landmarks in op.multi_hand_landmarks:
            draw.draw_landmarks(frame, all_landmarks, hands.HAND_CONNECTIONS)

            x = int(all_landmarks.landmark[8].x*frame_shape[1])
            y = int(all_landmarks.landmark[8].y*frame_shape[0])

            if prevxy!= None:
                cv2.line(mask, prevxy, (x,y), colour, thickness)
            prevxy = (x,y)
    #Merge frame and mask
    frame = np.where(mask, mask, frame)

    cv2.imshow('Live', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()



import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()
    img = cv2.flip(img,1 )

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            for id, lm in enumerate(handLms.landmark):
                # print("List: ", list(handLms.landmark))
                h, w, c = img.shape
                center_x , center_y = int(lm.x*w), int(lm.y*h)

                cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms)

    cv2.imshow("image", img)

    cv2.waitKey(1)
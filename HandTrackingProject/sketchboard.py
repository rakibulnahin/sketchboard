import cv2
import mediapipe as mp
from HandTrackingProject import HandTrackingModule as htm
import numpy as np

detector = htm.handDetector(detectionCon=0.85)

capture = cv2.VideoCapture(0)
x0, y0 = 0, 0
new_canvas = np.zeros((640, 640, 3), np.uint8)
color = (0, 0, 255)
thickness = 5

while True:
    _, img = capture.read()
    img = cv2.resize(img, (640, 640))
    img = cv2.flip(img, 1)

    # 1. Find Landmarks
    finding_hands_img = detector.findHands(img)
    finding_hands_list = detector.findPosition(finding_hands_img, draw=False)
    if len(finding_hands_list) > 0:
        # print(finding_hands_list)
    # tips of index and middle fingers
        x1, y1 = finding_hands_list[8][1:]
        x2, y2 = finding_hands_list[12][1:]


        # 2. Check for fingers up
        finger_list = detector.fingersUp()
        print(finger_list)

        # 5. Color Mode

        if finger_list[0]:
            # BLUE
            if finger_list[1] and finger_list[2] and finger_list[3]:
                color = (19, 12, 235)
                cv2.putText(img, "BLUE", org=(50, 50), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=color)
            # GREEN
            elif finger_list[3] and finger_list[2]:
                color = (0, 255, 0)
                cv2.putText(img, "GREEN", org=(50, 50), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=color)
            # RED
            elif finger_list[1] and finger_list[2]:
                color = (255, 0, 0)
                cv2.putText(img, "RED", org=(50, 50), fontScale=1, fontFace=cv2.FONT_HERSHEY_PLAIN, color=color)

    # 3. Selection Mode
        if finger_list[1] and finger_list[2]:
            # cv2.rectangle(img, (x1, y1-25), (x2, y2+25), (0, 0, 255), cv2.FILLED)
            print("Selection Mode")
            x0, y0, = x1, y1

        # 4. Drawing Mode
        if finger_list[1] and finger_list[2] == False:
            cv2.circle(img, (x1, y1), 15 , (0, 255, 0), cv2.FILLED)
            print("Drawing Mode")

            if x0 == 0 and y0 == 0:
                x0, y0 = x1, y1

            cv2.line(img, (x0, y0), (x1, y1), color, thickness=thickness)
            cv2.line(new_canvas, (x0, y0), (x1, y1), color, thickness=thickness)
            x0, y0 = x1, y1


        # 6. Clear Canvas
        if finger_list[0] and finger_list[1] and finger_list[3] and finger_list[4] and finger_list[2]:
            new_canvas = np.zeros((640, 640,3), np.uint8)

    # Setting the drawing in the image frame
    gray = cv2.cvtColor(new_canvas, cv2.COLOR_BGR2GRAY) # making a grayscale image
    _, extract_inverse_gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV) # extracts the black marks from the backround and inverses the black drawing to white
    extract_inverse_gray = cv2.cvtColor(extract_inverse_gray, cv2.COLOR_GRAY2BGR)

    # bitwise operation are good for masking the image parts from background
    # bitwise and is used for conjuntion of
    # simple doing binary bitwise opeartions of 0 and 1 and resulting images
    img = cv2.bitwise_and(img, extract_inverse_gray)
    img = cv2.bitwise_or(img, new_canvas) # bit wise or adds images of different in bit collection so adding the colored image in new canvase to new image of bitwise anded

    # img = cv2.addWeighted(img, 1, new_canvas, 1, 0)
    cv2.imshow("image", img)
    # cv2.imshow("new Canvase", new_canvas)
    # cv2.imshow("extract", extract_inverse_gray)
    k = cv2.waitKey(10)
    if k == ord("s"):
        break
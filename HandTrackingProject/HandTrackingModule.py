import cv2
import mediapipe as mp


class handDetector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]


    #find and draw hands
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw == True:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self, img, handNo=0, draw=True):

        self.lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                circle_x, circle_y = int(lm.x*w), int(lm.y*h)
                self.lm_list.append([id, circle_x, circle_y])
                if draw == True:
                    cv2.circle(img, (circle_x, circle_y), 15, (255,0, 0), cv2.FILLED)

        return self.lm_list


    def fingersUp(self):
        fingers = []

        # Check if fingers are found
        if len(self.lm_list) != 0:

            # Thumb
            if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers

            for i in range(1, 5):
                # print("First finger trail: ", "id: ", i, ": ", self.lm_list[self.tip_ids[i]][2])
                # print("Second finger trail: ", "id: ", i, ": ", self.lm_list[self.tip_ids[i]-2][2])
                if self.lm_list[self.tip_ids[i]][2] < self.lm_list[self.tip_ids[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

def main():
    capture = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    detector = handDetector()
    while True:
        _, img = capture.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img=img)
        lm_list = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[8])

        cv2.imshow("image", img)
        k = cv2.waitKey(1)
        if k == ord("s"):
            break

if __name__ == "__main__":
    main()
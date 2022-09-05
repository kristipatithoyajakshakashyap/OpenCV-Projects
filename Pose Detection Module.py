import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False,uBody = False,smooth=True,
                 detectionCon = 0.5,trackCon=0.5, modelComplexity = 1):
        self.mode = mode
        self.uBody = uBody
        self.smooth = smooth
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw  = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.modelComplex,self.mode,self.uBody,self.smooth,self.detectionCon,self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self, img , draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 6, (255,0,0), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    while True:
        sucess, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList)
            cv2.circle(img,(lmList[14][1], lmList[14][2]),15, (0,0,255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 2, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

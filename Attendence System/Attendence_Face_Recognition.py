import cv2
import numpy as np
import os
import face_recognition as fr
from datetime import datetime
path = 'ImageAttendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDateList = f.readlines()
        nameList = []
        for line in myDateList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            tstring = now.strftime('%H:%M:%S')
            dstring = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tstring},{dstring}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSize = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSize = cv2.cvtColor(imgSize, cv2.COLOR_BGR2RGB)

    faceCurFrame = fr.face_locations(imgSize)
    encodeCurFrame = fr.face_encodings(imgSize)

    for encodeFace, facLoc in zip(encodeCurFrame,faceCurFrame):
        matches =fr.compare_faces(encodeListKnown,encodeFace)
        faceDis = fr.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = facLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1), (x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)
    cv2.imshow('webcame',img)
    cv2.waitKey(1)


imgObama = fr.load_image_file('images/obama.jpg')
imgObama = cv2.cvtColor(imgObama, cv2.COLOR_BGR2RGB)
imgTest = fr.load_image_file('images/obama2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

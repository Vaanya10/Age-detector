import cv2
import argparse
import math
parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()
faceproto = "C:/Users/dines/Python folder/Age detector/opencv_face_detector.pbtxt"
facemodel = "C:/Users/dines/Python folder/Age detector/opencv_face_detector_uint8.pb"
ageproto = "C:/Users/dines/Python folder/Age detector/age_deploy.prototxt"
agemodel = "C:/Users/dines/Python folder/Age detector/age_net.caffemodel"
genderproto = "C:/Users/dines/Python folder/Age detector/gender_deploy.prototxt"
gendermodel = "C:/Users/dines/Python folder/Age detector/gender_net.caffemodel"

facenet = cv2.dnn.readNet(facemodel,faceproto)
agenet = cv2.dnn.readNet(agemodel,ageproto)
gendernet = cv2.dnn.readNet(gendermodel,genderproto)
video = cv2.VideoCapture(args.image if args.image else 0)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameheight = frameOpencvDnn.shape[0]
    framewidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn,1.0,(300,300),[104,117,127],True,False)
    net.setInput(blob)
    detections = net.forward()
    faceboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*framewidth)
            y1=int(detections[0,0,i,4]*frameheight)
            x2=int(detections[0,0,i,5]*framewidth)
            y2=int(detections[0,0,i,6]*frameheight)
            faceboxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn,(x1,y1),(x2,y2),(0,255,0),int(round(frameheight/150)))
    return frameOpencvDnn,faceboxes

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

padding = 20

while cv2.waitKey(1)<0:
    hasframe,frame = video.read()
    if not hasframe:
        cv2.waitKey()
        break
    resultimg,faceboxes = highlightFace(facenet,frame)
    if not faceboxes:
        print('No face Detected')
    for faceBox in faceboxes:
        face=frame[max(0,faceBox[1]-padding):
            min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):
            min(faceBox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        gendernet.setInput(blob)
        genderpredict=gendernet.forward()
        gender = genderList[genderpredict[0].argmax()]
        print(f'gender:{gender}')
        agenet.setInput(blob)
        agepredictor = agenet.forward()
        age = ageList[agepredictor[0].argmax()]
        print(f'age:{age[1:-1]} years')
        cv2.putText(resultimg,f'{gender},{age}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow('frame',resultimg)


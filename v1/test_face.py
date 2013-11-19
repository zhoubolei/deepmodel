from faceModule import faceDetector
import cv2.cv as cv

def testImg(faceXMLpath):
    #faceXMLpath =  "/home/bolei/code/opencv-2.4.7/data/haarcascades/haarcascade_frontalface_alt.xml"
    myDetector = faceDetector(faceXMLpath)
    for i in range(1,22):
        input_name = '/home/bolei/Pictures/face/' + str(i) + '.jpg'
        face_set, img_rectangle = myDetector.detectImg(input_name)
        if face_set!=0:
            print face_set
            cv.ShowImage("result", img_rectangle)
            cv.WaitKey()

def testCam(faceXMLpath):
    #faceXMLpath =  "/home/bolei/code/opencv-2.4.7/data/haarcascades/haarcascade_frontalface_alt.xml"
    myDetector = faceDetector(faceXMLpath)
    capture = myDetector.openCam()
    if capture:
        while 1:
            face_set, img_rectangle = retrieveCam()
            if face_set!=0:
                cv.ShowImage("result", img_rectangle)
    
if __name__ == '__main__':
    #testImg() # test input image
    #testCam() # test camera image
    #faceXMLpath =  "/home/bolei/code/opencv-2.4.7/data/haarcascades/haarcascade_frontalface_alt.xml"
    faceXMLpath = "/afs/csail.mit.edu/u/b/bzhou/code/OpenCV-2.4.2/data/haarcascades/haarcascade_frontalface_alt.xml"
    myDetector = faceDetector(faceXMLpath)
    capture = myDetector.openCam()
    while 1:
        face_set, img_rectangle = myDetector.retrieveCam()
        cv.ShowImage("result", img_rectangle)
        print face_set
        if cv.WaitKey(10) >= 0:
            break

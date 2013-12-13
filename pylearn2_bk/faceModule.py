### face detection module. 
### install OpenCV 2.4+

import sys
import cv2.cv as cv
from optparse import OptionParser
import numpy as np


min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

class faceDetector(object):
    def __init__(self, xmlpath): 
        parser = OptionParser(usage = "")
        parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = xmlpath)
        (options, args) = parser.parse_args()
        self.cascade = cv.Load(options.cascade)
    
    
    def detect_and_draw(self, img):
        # allocate temporary images
        gray = cv.CreateImage((img.width,img.height), 8, 1)
        small_img = cv.CreateImage((cv.Round(img.width / image_scale),
                       cv.Round (img.height / image_scale)), 8, 1)

        # convert color input image to grayscale
        cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

        # scale input image for faster processing
        cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

        cv.EqualizeHist(small_img, small_img)
        
        if (self.cascade):
            faces = cv.HaarDetectObjects(small_img, self.cascade, cv.CreateMemStorage(0),
                                         haar_scale, min_neighbors, haar_flags, min_size)
            if faces:
                face_set = []
                for ((x, y, w, h), n) in faces:
                    # the input to cv.HaarDetectObjects was resized, so scale the
                    # bounding box of each face and convert it to two CvPoints
                    img48 = cv.CreateImage((48, 48), 8, 1)
                    pt1 = (int(x * image_scale), int(y * image_scale))
                    pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                    img_face_small = small_img[y:y+h,x:x+w]
                    cv.Resize(img_face_small, img48, cv.CV_INTER_LINEAR)
                    face_vector = np.asarray(img48[:,:])
                    face_vector = face_vector.reshape(48*48)
                    face_set.append(face_vector)
                    cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)

                return [face_set, img]
            else:
            #cv.ShowImage("result", img)
                return [0,0]

    def detectImg(self, input_name):
        image = cv.LoadImage(input_name, 1)      
        face_set, img_rectangle = self.detect_and_draw(image)
        return [face_set, img_rectangle]
        
    def openCam(self):   
        self.capture = cv.CreateCameraCapture(0)
        return self.capture
        
    def retrieveCam(self):
        if self.capture:
            frame_copy = None
            frame = cv.QueryFrame(self.capture)
            if not frame:
                return 0
            if not frame_copy:
                frame_copy = cv.CreateImage((frame.width,frame.height),
                                            cv.IPL_DEPTH_8U, frame.nChannels)
            if frame.origin == cv.IPL_ORIGIN_TL:
                cv.Copy(frame, frame_copy)
            else:
                cv.Flip(frame, frame_copy, 0)
            face_set, img_rectangle = self.detect_and_draw(frame_copy)
            return face_set, img_rectangle
        else:
            return [0,0]
        
        self.capture = cv.CreateCameraCapture(2)
        return self.capture
        
    def retrieveCam(self):
        if self.capture!=0:        
            #frame_copy = None
            frame = cv.QueryFrame(self.capture)
            if not frame:
                return [0,0]
            #if not frame_copy:
            #    frame_copy = cv.CreateImage((frame.width,frame.height),cv.IPL_DEPTH_8U, frame.nChannels)
            #if frame.origin == cv.IPL_ORIGIN_TL:
            #    cv.Copy(frame, frame_copy)
            #else:
            #    cv.Flip(frame, frame_copy, 0)

            face_set, img_rectangle = self.detect_and_draw(frame)
            
            if face_set!=0:
                #cv.ShowImage("result", img_rectangle)
                return [face_set, img_rectangle]
            else:
                return [0, frame]
                #cv.ShowImage("result", frame_copy)


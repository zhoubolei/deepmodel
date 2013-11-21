import sys
import cv2.cv as cv
from optparse import OptionParser
import numpy as np

# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=2, min_neighbors=3, flags=0) are tuned
# for accurate yet slow object detection. For a faster operation on real video
# images the settings are:
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING,
# min_size=<minimum possible face size

min_size = (20, 20)
image_scale = 2
haar_scale = 1.2
min_neighbors = 2
haar_flags = 0

def detect_and_draw(img, cascade):
    # allocate temporary images
    gray = cv.CreateImage((img.width,img.height), 8, 1)
    small_img = cv.CreateImage((cv.Round(img.width / image_scale),
                   cv.Round (img.height / image_scale)), 8, 1)

    # convert color input image to grayscale
    cv.CvtColor(img, gray, cv.CV_BGR2GRAY)

    # scale input image for faster processing
    cv.Resize(gray, small_img, cv.CV_INTER_LINEAR)

    cv.EqualizeHist(small_img, small_img)

    img48 = cv.CreateImage((48, 48), 8, 1)
    if (cascade):
        t = cv.GetTickCount()
        faces = cv.HaarDetectObjects(small_img, cascade, cv.CreateMemStorage(0),
                                     haar_scale, min_neighbors, haar_flags, min_size)
        t = cv.GetTickCount() - t
        print "detection time = %gms" % (t/(cv.GetTickFrequency()*1000.))
        if faces:
            for ((x, y, w, h), n) in faces:
                # the input to cv.HaarDetectObjects was resized, so scale the
                # bounding box of each face and convert it to two CvPoints
                
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                #img_face = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
                img_face_small = small_img[y:y+h,x:x+w]
                cv.Resize(img_face_small, img48, cv.CV_INTER_LINEAR)
                face_vector = np.asarray(img48[:,:])
                face_vector = face_vector.reshape(48*48)
                print face_vector
                cv.Rectangle(img, pt1, pt2, cv.RGB(255, 0, 0), 3, 8, 0)
    if faces:
        cv.ShowImage("result", img48)
        return face_vector
    else:
        cv.ShowImage("result", img)
<<<<<<< HEAD
        return []
=======
        return 0
>>>>>>> 863a50da766d6ba8ce099e4e25cc4ff33587b5c3
        
if __name__ == '__main__':

    parser = OptionParser(usage = "")
    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = "/afs/csail.mit.edu/u/b/bzhou/code/OpenCV-2.4.2/data/haarcascades/haarcascade_frontalface_alt.xml")
    (options, args) = parser.parse_args()

    cascade = cv.Load(options.cascade)

    capture = cv.CreateCameraCapture(0)

    cv.NamedWindow("result", 1)

    if capture:
        frame_copy = None
        while True:
            frame = cv.QueryFrame(capture)
            if not frame:
                cv.WaitKey(0)
                break
            if not frame_copy:
                frame_copy = cv.CreateImage((frame.width,frame.height),
                                            cv.IPL_DEPTH_8U, frame.nChannels)
            if frame.origin == cv.IPL_ORIGIN_TL:
                cv.Copy(frame, frame_copy)
            else:
                cv.Flip(frame, frame_copy, 0)

            face_vector = detect_and_draw(frame_copy, cascade)

            if cv.WaitKey(10) >= 0:
                break

    cv.DestroyWindow("result")

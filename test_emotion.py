from cNetworkIO import DeepLearner
from facedetect_cam import *
 

if __name__ == '__main__':
    learner = DeepLearner()
    #learner.Test()
    faceXMLpath =  "/home/bolei/code/opencv-2.4.7/data/haarcascades/haarcascade_frontalface_alt.xml"
    parser = OptionParser(usage = "")
    parser.add_option("-c", "--cascade", action="store", dest="cascade", type="str", help="Haar cascade file, default %default", default = faceXMLpath)
    (options, args) = parser.parse_args()

    cascade = cv.Load(options.cascade)
    
    input_name = '/home/bolei/Pictures/face/1.jpg'
    image = cv.LoadImage(input_name, 1)
    face_vector = detect_and_draw(image, cascade)
    
    emotionTag = learner.TestSingle(face_vector)
    print emotionTag
    
    #self.tag = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
    cv.WaitKey(0)

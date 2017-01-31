import numpy as np
import cv2
import subprocess
import time
from face_detector2 import getFaces
from chainer import serializers
from face_detector2 import getFaces
from classify_new2 import test,VGGNet

import dlib

#from clasify import test
#model=VGGNet()
#serializers.load_hdf5("./face_recognition/VGG11_0223096959822.model",model)##input model path
#mean=np.load("./face_recognition/mean.npy")##input mean path

def extract(input_video):
    detector = dlib.get_frontal_face_detector()
    remove_mp4 = input_video.split('.',1)[0]
    def detect_face(image):
        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
        # Read the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        print "Found {0} faces!".format(len(faces))

        # Draw a rectangle around the faces
    #	for (x, y, w, h) in faces:
    #		cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        results=[]
        for (x, y, w, h) in faces:
            results.append(cv2.resize(image[y : y+h, x : x+w],(96,96)))
        #len(faces) show the number of face. if len(faces)>0, then extract
        #in returned image, face is surrounded by box
        return results, len(faces)

    model=VGGNet()
    model.to_gpu()
    serializers.load_hdf5("./face_recognition/VGGface_00010659227118.model",model)##input model path
    mean=np.load("./face_recognition/mean.npy")##input mean path
    frame_skip = 10 


    output = 'ev1_'+input_video
    
    input_video = './video/'+ input_video 
    output = './video/'+ output
    print input_video
    print output
    extend_frame = 50 * frame_skip #extend_frame must be larger than frame_skip
    #movie frame
    frame_s=[]#start frame of extraction
    frame_e=[]#end frame of extraction
    frame_number = -1# frame number of the movie
    frame_fps = -1#frame per second
    minimum_duration=0#the minimum of play time of one extracted movie 


    def export_sparse_joint():
        cap = cv2.VideoCapture(input_video)#input video
        cmd="ffmpeg -i "+input_video+" 2>&1 | grep Duration | awk '{print $2}' | tr -d ,"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        std_out, std_err = p.communicate()
        play_time=std_out.split(":")
        tot_play_time=float(play_time[0])*60*60+float(play_time[1])*60+float(play_time[2])#total time of play
        frame_number = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    #	frame_number = 300
        frame_fps = frame_number/tot_play_time
    #	frame_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        # form of output 
    #	fourcc = cv2.cv.CV_FOURCC(*'XVID')
    #	size = ((int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        # open output
    #	out = cv2.VideoWriter("./output1.avi", fourcc, frame_fps, size)

        before_i=-1*frame_skip
        after_i=-1*frame_skip
        frame_s.append(0)
        for i in xrange(frame_number):
            ret,frame = cap.read()
            print "{0} / {1}".format(i,frame_number)
            if (i%frame_skip) == 0 and ret == True:
    #
    #
    #here use the detector and the clasifier
    #
    #
                results, ___=getFaces(frame,detector)
                len_faces = len(results)
                #results, len_faces = detect_face(frame)
                if len_faces > 0:
                    ret, _, _, _ = test(results,model=model, mean=mean)
                    if ret:
                        before_i=after_i
                        after_i=i
                        if ((after_i - before_i) > extend_frame):
                                frame_e.append(before_i)
                                frame_s.append(after_i)
        frame_e.append(after_i)
        cap.release()
    #	out.release()
        #cv2.destroyAllWindows()
        nigehaji_num=0
        for i in xrange(len(frame_s)):
            duration = float (frame_e[i] - frame_s[i])
            if duration > minimum_duration:
                duration /= frame_fps
                start = frame_s[i] / frame_fps
                cmd = "ffmpeg -ss " + str(start) + " -i " + input_video + " -t "+ str(duration) + " -vcodec copy -acodec copy ./video/after_remove" + remove_mp4 + str(nigehaji_num) + ".mp4"
                subprocess.call(cmd, shell=True)
                nigehaji_num+=1
        cmd = "ffmpeg "
        for i in xrange(nigehaji_num):
            cmd = cmd + "-i ./video/after_remove" + remove_mp4 +str(i)+ ".mp4 "
        cmd = cmd + "-strict -2 -filter_complex 'concat=n="+str(nigehaji_num)+":v=1:a=1' " + output
        subprocess.call(cmd, shell=True)
        print nigehaji_num
        cmd = "rm ./video/after_remove*"
        subprocess.call(cmd,shell=True)
    export_sparse_joint()
    return output

if __name__ == '__main__':
    input_video = 'nigehajikai.mp4'
    output=extract(input_video)
    print output


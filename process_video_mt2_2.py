import numpy as np
import cv2
import subprocess
import time
from face_detector2 import getFaces
from chainer import serializers
from collections import Counter
import dlib
#from clasify2 import test
#from clasify2 import VGGNet
#model=VGGNet()
#serializers.load_hdf5("./face_recognition/VGG11_00226621233099.model",model)##input model path
#mean=np.load("./face_recognition/mean.npy")##input mean path
from classify_new2 import test,VGGNet
from multiprocessing import Pool
model=VGGNet()
model.to_gpu()
serializers.load_hdf5("./face_recognition/VGGface_00010659227118.model",model)##input model path
mean=np.load("./face_recognition/mean.npy")##input mean path
nantoka = 'nigehajikai'
input_video = './video/'+nantoka+'.mp4'
#skip=10#if you want otoarimovie, fps / skip must be integer
testnum = 0
otoarimovie = True
create_video = True
collect_face=False
now_time=time.time()
output1 = "./video/pvbws"+nantoka+str(now_time)+".avi"
output2 = "./video/pvbws"+nantoka+str(now_time)+".mp4"
who_are_there = []
thread_num = 8
size = (0,0)
detector = dlib.get_frontal_face_detector()
def export_movie():
    #target movie
    global size
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    size = ((int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fps = 8
    frame_number = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))*fps//int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    print "fps={0}".format(fps)
    print "frame_number={0}".format(frame_number)
    # open output
    out = cv2.VideoWriter(output1, fourcc, fps, size)
    print frame_number
    i = 0
    #for i in xrange(frame_number):
    p = Pool(thread_num)
    while cap.isOpened():
        frames = []
        for j in range(thread_num):
            cap.set(0,1000//fps*i)
            i+=1
            print "{0} / {1}".format(i,frame_number)
            if i<frame_number:
                frames.append(cap.read())
            else:
                frames.append([False,None])
        processed = p.map(process,frames)
        for count,[exist, results,leftbottoms,frame] in enumerate(processed):
            # write the flipped frame
            proba = np.array([])
            if results!=None and len(results) > 0:
                ret2, strings, prediction,proba = test(results, model, mean)
                [cv2.putText(frame, text=strings[j], org=leftbottoms[j], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2) for j in range(len(prediction))]
                [cv2.imwrite("./testdazo/"+strings[j]+"f"+str(i)+"k"+str(j)+".jpg", results[j]) for j in range(len(prediction)) if collect_face]
            else:
                cv2.putText(frames[count][1], text="There Is None", org=(size[0]/3,size[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0,0,255), thickness=2)
                prediction = np.array([],dtype=np.int32)




            
            if create_video and results!=None:
                out.write(frame)
                who_are_there.append(prediction[np.where(proba>0.9)[0]])
        if not frames[-1][0]:
            break
        #cv2.imshow('frame1',processed[0][2])
        #k = cv2.waitKey(1)
    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    
    if otoarimovie:
        cmd='ffmpeg -i '+output1+' -i '+input_video+' -vcodec copy -acodec copy '+output2
        subprocess.call(cmd, shell=True)
    
    return who_are_there
def process(ret1_frame):
    ret1,frame = ret1_frame
    if ret1:
        #results = detect_face(frame)
        results, leftbottoms = getFaces(frame,detector)
        return True,results,leftbottoms,frame
    else:
        return False,None,None,None
    
if __name__ == '__main__':
    start = time.time()
    wat = export_movie()
    np.save("wat.npy",np.array(wat))
    #print wat
    print time.time() - start

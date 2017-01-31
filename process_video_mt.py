import numpy as np
import cv2
import subprocess
import time
from face_detector2 import getFaces
from chainer import serializers
from collections import Counter
#from clasify2 import test
#from clasify2 import VGGNet
#model=VGGNet()
#serializers.load_hdf5("./face_recognition/VGG11_00226621233099.model",model)##input model path
#mean=np.load("./face_recognition/mean.npy")##input mean path
from clasify3 import test
from clasify3 import VGGNet
from multiprocessing import Pool
model=VGGNet()
serializers.load_hdf5("./face_recognition/VGG11_0223096959822.model",model)##input model path
model.to_gpu()
mean=np.load("./face_recognition/mean.npy")##input mean path
nantoka = 'nigehaji'
input_video = './'+nantoka+'.mp4'
#skip=10#if you want otoarimovie, fps / skip must be integer
testnum = 0
otoarimovie = True
create_video = True
collect_face=False
output1 = "./pvbws"+nantoka+str(otoarimovie)+str(testnum)+".avi"
output2 = "./pvbws"+nantoka+str(otoarimovie)+str(testnum)+".mp4"
who_are_there = []
thread_num = 7
size = (0,0)
def export_movie():
    #target movie
    global size
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    size = ((int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), (int)(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fps = 8
    frame_number = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))*fps//60
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
        for exist, who,frame in processed:
            # write the flipped frame
            if create_video and exist:
                out.write(frame)
                who_are_there.append(who)
        if not frames[-1][0]:
            break
        #cv2.imshow('frame1',processed[0][2])
        k = cv2.waitKey(1)
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
        results, leftbottoms = getFaces(frame)
        if len(results) > 0:
            ret2, strings, prediction = test(results, model, mean)
            print "prediction",prediction
            [cv2.putText(frame, text=strings[j], org=leftbottoms[j], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2) for j in range(len(prediction))]
            def gen(prediction):
                predictionsorted = sorted(prediction)
                for k in range(0,len(predictionsorted)-1):
                    for l in range(k+1,len(predictionsorted)):
                        print (predictionsorted[k], predictionsorted[l])
                        yield (predictionsorted[k], predictionsorted[l])
                
            [cv2.imwrite("./testdazo/"+strings[j]+"f"+str(i)+"k"+str(j)+".jpg", results[j]) for j in range(len(prediction)) if collect_face]
        else:
            cv2.putText(frame, text="There Is None", org=(size[0]/3,size[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(0,0,255), thickness=2)
            prediction = np.array([])
        return True,prediction,frame
    else:
        return False,None,None
    
if __name__ == '__main__':
    start = time.time()
    wat = export_movie()
    print wat
print time.time() - start

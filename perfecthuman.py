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
import dlib
detector = dlib.get_frontal_face_detector()
def export_movie(input_video):
    global detector
    model=VGGNet()
    model.to_gpu()
    serializers.load_hdf5("./face_recognition/VGGface_00010659227118.model",model)##input model path
    mean=np.load("./face_recognition/mean.npy")##input mean path
    #skip=10#if you want otoarimovie, fps / skip must be integer
    testnum = 0
    extract_video = True
    create_video = True
    collect_face=False
    remove_mp4 = input_video.split('.',1)[0]
    output1 = "./video/pv_"+remove_mp4+".avi"
    output2 = "./video/pv_"+remove_mp4+".mp4"
    output3 = "./video/ev_"+remove_mp4+".mp4"
    input_video = './video/'+ input_video 
    who_are_there = []
    frame_s=[]#start frame of extraction
    frame_e=[]#end frame of extraction
    before_i=-1
    after_i=-1
    thread_num = 8
    size = (0,0)
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
    minimum_duration=0#s
    extend_frame = 10#>1 or =1
    before_i=-1
    after_i=-1
    frame_s.append(0)
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
                if extract_video:
                    if ret2:
                        before_i=after_i
                        after_i=i - thread_num + count
                        if ((after_i - before_i) > extend_frame):
                                frame_e.append(before_i)
                                frame_s.append(after_i)
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
    print frame_s
    print frame_e
    #cv2.destroyAllWindows()
    frame_e.append(after_i)
    cmd='ffmpeg -i '+output1+' -i '+input_video+' -vcodec copy -acodec copy '+output2
    subprocess.call(cmd, shell=True)
    if extract_video:
        nigehaji_num=0
        for i in xrange(len(frame_s)):
            duration = (float)(frame_e[i] - frame_s[i])
            duration /= fps
            print duration
            if duration > minimum_duration:
                start = frame_s[i] / fps
                cmd = "ffmpeg -ss " + str(start) + " -i " + input_video + " -t "+ str(duration) + " -vcodec copy -acodec copy ./video/after_remove" + remove_mp4 + str(nigehaji_num) + ".mp4"
                subprocess.call(cmd, shell=True)
                nigehaji_num+=1
        cmd = "ffmpeg "
        for i in xrange(nigehaji_num):
            cmd = cmd + "-i ./video/after_remove" + remove_mp4 +str(i)+ ".mp4 "
        cmd = cmd + "-strict -2 -filter_complex 'concat=n="+str(nigehaji_num)+":v=1:a=1' " + output3
        subprocess.call(cmd, shell=True)
        print nigehaji_num
        cmd = "rm ./video/after_remove*"
        subprocess.call(cmd,shell=True)
    np.save("./visualization-master/wat_"+remove_mp4+".npy",np.array(who_are_there))
    
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
    input_video='2minutes.mp4'
    remove_mp4 = input_video.split('.',1)[0]
    wat = export_movie(input_video)
    #print wat
    print time.time() - start

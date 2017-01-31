
import socket
import numpy as np
import cv2
import re
import time
import os
import chainer
import glob
from chainer import cuda, Variable, FunctionSet, optimizers, serializers 
import chainer.functions  as F
import chainer.links as L
from PIL import Image
import time
from face_detector2 import getFaces

class VGGNet(chainer.Chain):
    """
    VGGNet
    - It takes (96, 96, 3) sized image as imput
    """
    
    def __init__(self):
        super(VGGNet, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            
            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            
            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            
            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            
            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            
            fc6=L.Linear(3*3*512, 200),
            #fc7=L.Linear(512, 200),
            fc8=L.Linear(200,2)
        )
        
    def __call__(self, x, t,train):
        start=time.time()
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        print time.time()-start
        start=time.time()
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        print time.time()-start
        start=time.time()
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        print time.time()-start
        start=time.time()
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        print time.time()-start
        start=time.time()
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        print time.time()-start
        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        #h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)s
        h = self.fc8(h)
        
        self.loss = F.softmax_cross_entropy(h, t)
        self.acc = F.accuracy(h, t)
        self.pred = F.softmax(h)
        return self.loss,self.acc,self.pred

model=VGGNet()
serializers.load_hdf5("./face_recognition/VGG11_00226621233099.model",model)##input model path
mean=np.load("./face_recognition/mean.npy")##input mean path

def test(images,model,mean):

    X=[]
    for im in images:
        X.append(np.transpose(cv2.resize(im,(96,96)),(2,0,1))/255.)
    t=np.zeros(len(X))

    X=np.array(X,dtype=np.float32)
    t=np.array(t,dtype=np.int32)
    X-=mean
    
    _,_,pre = model(X,t,train=False)
    classes=["aragaki_ari","no_aragaki"]
    prediction=np.argmax(pre.data,axis=1)
    probability=np.max(pre.data,axis=1)

    for i,a in enumerate(prediction):
        print(classes[a]+"    "+str(probability[i]*100)+ "%")
    b=np.sum(prediction==0)
    if b==0:
        return False
    else:
        return True
if __name__ == '__main__':
    lena, ___=getFaces(cv2.imread('lena.jpeg'))
    test(lena, model,mean)





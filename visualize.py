#--encoding=utf-8--
from igraph import *
import numpy as np
from numpy import array
from numpy import float64

vertices = ["新垣", "星野","成田", "藤井","真野","大谷", "山賀","石田"]

window_size=40
min_cofreq=10

m=[(i,j) for i in range(len(vertices)) for j in range(i+1,len(vertices))]
dic={}
for i in range(len(m)):
     dic[i]=m[i]
a=np.load("./visualization-master/name_sequence.npy")
ara=np.array([i for i,val in enumerate(a) if np.sum(val==0)!=0])
hoshi=np.array([i for i,val in enumerate(a) if np.sum(val==1)!=0])
narita=np.array([i for i,val in enumerate(a) if np.sum(val==2)!=0])
hujii=np.array([i for i,val in enumerate(a) if np.sum(val==4)!=0])
mano=np.array([i for i,val in enumerate(a) if np.sum(val==5)!=0])
ohtani=np.array([i for i,val in enumerate(a) if np.sum(val==6)!=0])
yamaga=np.array([i for i,val in enumerate(a) if np.sum(val==7)!=0])
ishida=np.array([i for i,val in enumerate(a) if np.sum(val==8)!=0])
ara=np.array([1 if np.sum((ara>=i) & (ara<i+window_size))>0 else 0 for i in range(len(a)-window_size)])
hoshi=np.array([1 if np.sum((hoshi>=i) & (hoshi<i+window_size))>0 else 0 for i in range(len(a)-window_size)]) 
narita=np.array([1 if np.sum((narita>=i) & (narita<i+window_size))>0 else 0 for i in range(len(a)-window_size)]) 
hujii=np.array([1 if np.sum((hujii>=i) & (hujii<i+window_size))>0 else 0 for i in range(len(a)-window_size)]) 
mano=np.array([1 if np.sum((mano>=i) & (mano<i+window_size))>0 else 0 for i in range(len(a)-window_size)]) 
ohtani=np.array([1 if np.sum((ohtani>=i) & (ohtani<i+window_size))>0 else 0 for i in range(len(a)-window_size)]) 
yamaga=np.array([1 if np.sum((yamaga>=i) & (yamaga<i+window_size))>0 else 0 for i in range(len(a)-window_size)]) 
ishida=np.array([1 if np.sum((ishida>=i) & (ishida<i+window_size))>0 else 0 for i in range(len(a)-window_size)]) 

mat=np.vstack((ara,hoshi,narita,hujii,mano,ohtani,yamaga,ishida))
co=np.dot(mat,mat.T)
ara_w=np.sum(co[:,0])>0
hoshi_w=np.sum(co[:,1])>0
narita_w=np.sum(co[:,2])>0
hujii_w=np.sum(co[:,3])>0
mano_w=np.sum(co[:,4])>0
ohtani_w=np.sum(co[:,5])>0
yamaga_w=np.sum(co[:,6])>0
ishida_w=np.sum(co[:,7])>0
tri=[list(v[i+1:]) for i,v in enumerate(co[:-1])]
seq=[]

for i in tri:
     seq.extend(i)
seq=np.array(seq).astype(np.float32)###(0,1),(0,2)...(6,7)の値
indices=np.where(seq>=min_cofreq)###入れたいseqのindex
tmp=np.argsort(seq)[::-1]###seqを大きい順に並べたindex
edges=[dic[kkkk] for kkkk in tmp if kkkk in indices[0]]
correlation=20*seq[tmp]/max(seq[tmp])+3
freq=np.diag(co).astype(np.float32)
g = Graph(edges=edges, directed=False)
color=["#ff9999","#0000ff","#00ff00","#00ffff","#ff00ff","#ff0000","#ffff00","#ffffff"]
label_color=["#006666","#ffff00","#ff00ff","#ff0000","#00ff00","#00ffff","#0000ff","#000000"]
edge_color="#ff9900"
layout=g.layout("kk")
visual_style={}
visual_style["vertex_size"] = 0
visual_style["vertex_label_color"]=label_color
visual_style["edge_width"] = correlation
visual_style["edge_color"] = [edge_color,edge_color,edge_color,edge_color,edge_color,edge_color,edge_color,edge_color]
visual_style["layout"] = layout
visual_style["vertex_color"]=color
visual_style["vertex_width"]=[0,0,0,0,0,0,0,0]
visual_style["margin"] = 20
plot(g, "./visualization-master/nige_correlation.png",**visual_style)
print([i for i in layout])

from PIL import Image
size=40+freq/max(freq)*70
ara=Image.open("./visualization-master/aragaki0.jpg").resize((size[0],size[0]))
hoshi=Image.open("./visualization-master/hoshino0.jpg").resize((size[1],size[1]))
narita=Image.open("./visualization-master/narita11.jpg").resize((size[2],size[2]))
fujii=Image.open("./visualization-master/fujii1.jpg").resize((size[3],size[3]))
mano=Image.open("./visualization-master/mano178.jpg").resize((size[4],size[4]))
ohtani=Image.open("./visualization-master/ohtani0.jpg").resize((size[5],size[5]))
yamaga=Image.open("./visualization-master/yamaga67.jpg").resize((size[6],size[6]))
ishi=Image.open("./visualization-master/ishi.jpeg").resize((size[7],size[7]))
b=Image.open("./visualization-master/nige_correlation.png")
a=np.array([i for i in layout])
print(a)
min_x=np.min(a[:,0],axis=0)
min_y=np.min(a[:,1],axis=0)
max_x=np.max(a[:,0],axis=0)
max_y=np.max(a[:,1],axis=0)
print(np.array(a[:,0]).reshape((-1,1)))

a[:,0]=(np.array(a[:,0])-min_x)/(max_x-min_x)*600
a[:,1]=(np.array(a[:,1])-min_y)/(max_y-min_y)*600
print(a)
a=[(int(ll[0]+100-size[i]/2),int(ll[1]+100-size[i]/2)) for i,ll in enumerate(a)]
print(a[0])

bg= Image.new("RGB",(800, 800),(255,255,255))
bg.paste(b,(100,100))
ara_w=np.sum(co[:,0])>0
hoshi_w=np.sum(co[:,1])>0
narita_w=np.sum(co[:,2])>0
hujii_w=np.sum(co[:,3])>0
mano_w=np.sum(co[:,4])>0
ohtani_w=np.sum(co[:,5])>0
yamaga_w=np.sum(co[:,6])>0
ishida_w=np.sum(co[:,7])>0
if ara_w:
     bg.paste(ara,a[0])
if hoshi_w:
     bg.paste(hoshi,a[1])
if narita_w:
     bg.paste(narita,a[2])
if hujii_w:
     bg.paste(fujii,a[3])
if mano_w:
     bg.paste(mano,a[4])
if ohtani_w:
     bg.paste(ohtani,a[5])
if yamaga_w:
     bg.paste(yamaga,a[6])
if ishida_w:
     bg.paste(ishi,a[7])
bg.save("./visualization-master/tmp.jpg")

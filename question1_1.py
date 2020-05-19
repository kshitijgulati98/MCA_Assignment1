#colour_autocorrelogram
import cv2
import numpy as np
import scipy as sp
from PIL import Image
import os
import pandas as pd

def distance(m,n,x,y,D):
    N1=[x+D,y+D]
    N2=[x-D,y-D]
    N3=[x+D,y]
    N4=[x-D,y]
    N5=[x,y+D]
    N6=[x,y-D]
    N7=[x-D,y+D]
    N8=[x+D,y-D]
    
    D_8=[N1,N2,N3,N4,N5,N6,N7,N8]
    abc=[]
    
    for i in D_8:
        if 0<=i[0]<m and 0<=i[1]<n:
            abc.append(i)
            
    return abc

def quantise_image(img,n):
    img=img.quantize(n)
    X,Y=img.size
    img=img.resize((int(X/2),int(Y/2)),resample=0)
    img=np.array(img)
    return img 


def auto_corr(image,num):
    image=Image.open(image)
    corr=[]
    
    image=quantise_image(image,num)
    m,n= image.shape
    
    distance_vector=[1,3,5,7]
    
    for i in distance_vector:
        count=0;
        ans=np.zeros(num)
            
        for x in range(m):
            for y in range(n):
                pixelval=image[x][y]
                
                nbors=distance(m,n,x,y,i)
                for j in nbors:
                    neighbour_val=image[j[0]][j[1]]
                    
                    for k in range(len(ans)):
                        if pixelval==neighbour_val:
                            count+=1
                            ans[pixelval]+=1
        
        vecx=np.ones(num)
        
        for i in range(num):
            if count!=0:
                vecx[i]=float(ans[i])/float(count)
            else:
                vecx[i]=0
        
        corr.append(vecx)
        
    return (corr)
                        


def corr_similarity(corr1,corr2,n):
    
    sum=[]
    for i in range(len(corr1)):
        for j in range(len(corr1[0])):
            num=abs(corr1[i][j]-corr2[i][j])
            den=1 + corr1[i][j] + corr2[i][j]
            val=float(num)/float(den)
            sum.append(val)
            
    sim=0;
    for i in sum:
        sim=sim+i
    sim=sim/float(n)
    return sim



if __name__ == "__main__":
              
     
    CC= pd.DataFrame(columns=['filename', 'CC'])
    
    directory="/Users/kshitij/Desktop/MCA/HW-1/images"
    for file in os.listdir(directory):
        if file.endswith(".jpg"): 
            autocc=auto_corr(file,64)
            print(file)
            print(autocc)
            new_row={"filename":file, "CC": autocc}
            CC=CC.append(new_row,ignore_index=True)
            continue
        else:
            continue
    pickle.dump(CC,open("cc.p","rb"))
    query_path = "./train/query"
    que, good, ok, junk = (0, 0, 0, 0)

    for i in os.listdir(query_path):
        que += 1
        names = list()
        score = list()
        comp = list()
        f = open(query_path+'/'+i)
        x = ''
        x = f.readline()
        x=x.split()[0]
        x=x[5:]
        y = pickle.load(open('./cc/'+x +'.p', 'rb'))

        for j in os.listdir('./cc'):
            if (j.split('.')[0] != x) :
                z = pickle.load(open('./cc/'+j , 'rb'))
                names.append(j.split('.')[0])
                score.append(corr_similarity(y,z,64))
        results = pd.DataFrame({'name': names ,'score':score})
        results = results.sort_values('score')

        truth = ['good','ok', 'junk']
            for a in truth:
            comp = list()
            jk = i.split('query')
            path = jk[0]
            ss = path + a
            file_f = open('./train/ground_truth/'+ ss +'.txt')
            for q in file_f :
                comp.append(q.split('\n')[0])
            num = np.shape(comp)
            check = list(results['name'][0:num[0]])
            for qq in comp :
                if qq in check and a == 'good':
                    good = good + 1
                elif qq in check and a == 'ok':
                    ok = ok + 1
                elif qq in check and a == 'junk':
                    junk = junk + 1

    print('good:',str(good/que))
    print('ok:'+ str(ok/que))
    print('junk:'+ str(junk/que))
        
    

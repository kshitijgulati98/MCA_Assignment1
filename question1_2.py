




#LOG
import cv2
import numpy as np
import os
import pandas as pd


def Blob(image):
    image=cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
    M,N=image.shape
    image=cv2.resize(image,(int(M/(2)),int(N/(2))),interpolation=cv2.INTER_CUBIC)
    
    k=1
    
    images=[]
    for i in range(1,11):
        sigma=k*i
        G= cv2.GaussianBlur(image, (0,0), sigmaX = sigma)
        L = cv2.Laplacian(G, cv2.CV_64F)
        images.append(L)
        
    space=np.stack(images)   
    
    Z,m,n=space.shape
    keypoints=list()
    
    for z in range(Z):
        for x in range(1,m):
            for y in range(1,n):
                #checking for all 26 neighbours
                    val0=pixel(x,y,z,m,n,Z,space)
                    val1=pixel(x-1,y-1,z,m,n,Z,space)
                    val2=pixel(x+1,y+1,z,m,n,Z,space)
                    val3=pixel(x-1,y+1,z,m,n,Z,space)
                    val4=pixel(x+1,y-1,z,m,n,Z,space)
                    val5=pixel(x,y-1,z,m,n,Z,space)
                    val6=pixel(x,y+1,z,m,n,Z,space)
                    val7=pixel(x+1,y,z,m,n,Z,space)
                    val8=pixel(x-1,y,z,m,n,Z,space)
                    va0=pixel(x,y,z+1,m,n,Z,space)
                    va1=pixel(x-1,y-1,z+1,m,n,Z,space)
                    va2=pixel(x+1,y+1,z+1,m,n,Z,space)
                    va3=pixel(x-1,y+1,z+1,m,n,Z,space)
                    va4=pixel(x+1,y-1,z+1,m,n,Z,space)
                    va5=pixel(x,y-1,z+1,m,n,Z,space)
                    va6=pixel(x,y+1,z+1,m,n,Z,space)
                    va7=pixel(x+1,y,z+1,m,n,Z,space)
                    va8=pixel(x-1,y+1,z+1,m,n,Z,space)
                    v0=pixel(x,y,z-1,m,n,Z,space)
                    v1=pixel(x-1,y-1,z-1,m,n,Z,space)
                    v2=pixel(x+1,y+1,z-1,m,n,Z,space)
                    v3=pixel(x-1,y+1,z-1,m,n,Z,space)
                    v4=pixel(x+1,y-1,z-1,m,n,Z,space)
                    v5=pixel(x,y-1,z-1,m,n,Z,space)
                    v6=pixel(x,y+1,z-1,m,n,Z,space)
                    v7=pixel(x+1,y,z-1,m,n,Z,space)
                    v8=pixel(x-1,y,z-1,m,n,Z,space)
                    
                    neighbourhood=[val1,val2,val3,val4,val5,val6,val7,val8,va0,va1,va2,va3,va4,va5,va6,va7,va8,v0,v1,v2,v3,v4,v5,v6,v7,v8]
                    while None in neighbourhood:
                        neighbourhood.remove(None)
                        
                    maxima=max(neighbourhood)
                    if val0>=maxima and val0>=0.03:
                        keypoints.append([x,y,1.414*(z+1)])
            


    return keypoints






def pixel(x,y,z,M,N,L,stack):
    try:
        if 0<x<=M and 0<=y<=N and 0<=z<=L:
            return stack[z][x][y]
        else :
            return None
    except IndexError as error:
        return None
    
    
    
    
 
if __name__ == "__main__":

    LOG= pd.DataFrame(columns=['filename', 'Blobs'])
    
    directory="/Users/kshitij/Desktop/MCA/HW-1/images"
    for file in os.listdir(directory):
        if file.endswith(".jpg"): 
            ans=Blob(file)
            print(file)
            print(ans)
            new_row={"filename":file, "Blobs": ans}
            LOG=LOG.append(new_row,ignore_index=True)
            LOG.to_csv(r'Blob.csv')
            continue
        else:
            continue







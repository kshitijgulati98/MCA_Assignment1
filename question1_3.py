#SURF
import cv2
import numpy as np
import scipy as sp
from PIL import Image
import math

from matplotlib import pyplot as plt

def DOG(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.
    M,N=img.shape
    no_octaves=4
    no_levels=5
    octaves=list()
    for i in range(no_octaves):
        octaves.append(cv2.resize(img,(int(M/(2**i)),int(N/(2**i))),interpolation=cv2.INTER_CUBIC))
    k=2**(1/3)
    sigma = 1.6
    DOG=list()
    for image in octaves:
        G0= cv2.GaussianBlur(image, (0,0), sigmaX =sigma*(k**(0)))
        G1= cv2.GaussianBlur(image, (0,0), sigmaX =sigma*(k**(1)))
        G2= cv2.GaussianBlur(image, (0,0), sigmaX =sigma*(k**(2)))
        G3= cv2.GaussianBlur(image, (0,0), sigmaX =sigma*(k**(3)))
        G4= cv2.GaussianBlur(image, (0,0), sigmaX =sigma*(k**(4)))
        L0=G1-G0
        L1=G2-G1
        L2=G3-G2
        L3=G4-G3
        space=[L0,L1,L2,L3]
        st=np.stack(space,axis=0)
        DOG.append(st)
        
    return DOG
  

#DOG returns 4 octaves(each with a DOG space of 4 levels)
                             
def hessian(x):
#https://stackoverflow.com/questions/31206443/numpy-second-derivative-of-a-ndimensional-array
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) 
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

#function to ectract keypoints in each octave
#https://www.ipol.im/pub/art/2014/82/article_lr.pdf
def keypoints(DOG):
    ans=list()
    for octave in DOG:
        list_of_keypoints=list()
        Z,X,Y=octave.shape
        
        for z in range(1,3): # can only see maxima in the second and third level as there is no levels above and..
            #below the 
            hessianx=hessian(octave[z,:,:])
            for x in range(1,X-1):
                for y in range(1,Y-1):
                    hessianpoint=hessianx[:,:,x,y]
                    val0=octave[z][x][y]
                    val1=octave[z][x-1][y-1]
                    val2=octave[z][x+1][y+1]
                    val3=octave[z][x-1][y+1]
                    val4=octave[z][x+1][y-1]
                    val5=octave[z][x][y-1]
                    val6=octave[z][x][y+1]
                    val7=octave[z][x+1][y]
                    val8=octave[z][x-1][y]
                    va0=octave[z+1][x][y]
                    va1=octave[z+1][x-1][y-1]
                    va2=octave[z+1][x+1][y+1]
                    va3=octave[z+1][x-1][y+1]
                    va4=octave[z+1][x+1][y-1]
                    va5=octave[z+1][x][y-1]
                    va6=octave[z+1][x][y+1]
                    va7=octave[z+1][x+1][y]
                    va8=octave[z+1][x-1][y]
                    v0=octave[z-1][x][y]
                    v1=octave[z-1][x-1][y-1]
                    v2=octave[z-1][x+1][y+1]
                    v3=octave[z-1][x-1][y+1]
                    v4=octave[z-1][x+1][y-1]
                    v5=octave[z-1][x][y-1]
                    v6=octave[z-1][x][y+1]
                    v7=octave[z-1][x+1][y]
                    v8=octave[z-1][x-1][y]
                    
                    neighbourhood=[val1,val2,val3,val4,val5,val6,val7,val8,va0,va1,va2,va3,va4,va5,va6,va7,va8,v0,v1,v2,v3,v4,v5,v6,v7,v8]
                    maxval=max(neighbourhood)
                    edgethresh=10
                    edgenessval=float(((edgethresh+1)**2)/edgethresh)
                    
                    trace=np.trace(hessianpoint, offset=0)
                    num=np.square(trace)
                    den=np.linalg.det(hessianpoint)
                    edgeness=float(num)/float(den)
                    if val0>=maxval and val0>=0.03 and edgeness<edgenessval:
                        list_of_keypoints.append([z,x,y])
            
        ans.append(list_of_keypoints)

    return ans
                    
           
      

def orientation_assignment(keypoints,image):
    bins=36
        
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.
    M,N=image.shape
    no_octaves=4
    no_levels=5
    images=list()
    for i in range(no_octaves):
        images.append(cv2.resize(image,(int(M/(2**i)),int(N/(2**i))),interpolation=cv2.INTER_CUBIC))
    
    orientations_octave=list()
    
    for i in range(len(keypoints)):
        img=images[i]
        no_keypoints=len(keypoints[i]) #no of keypoints per octave
        orientations=list()
        for j in range(no_keypoints):
            keypoint=keypoints[i][j]
            scale=keypoint[0]
            window=int(2*scale)+ 1
            x=keypoint[1]
            y=keypoint[2]
            L=cv2.GaussianBlur(img, (window,window), sigmaX =1.5*scale)
            mag= math.sqrt((L[x+1][y] - L[x-1][y])**2 + (L[x][y+1]-L[x][y-1])**2) 
            theta=(L[x][y+1] - L[x][y-1])/(L[x+1][y]-L[x-1][y])
            angle=math.degrees(np.arctan(theta))
            orientations.append([x,y,scale,mag,theta])
        orientations_octave.append(orientations)

    return orientations,len(orientations)
            
        
    

def Descriptor(image):
    image=cv2.imread(image)
    
    key=DOG(image)
    keyp=keypoints(key)
    ab=orientation_assignment(keyp,image)  
    return ab

if __name__ == "__main__":
    
    
    
    surf= pd.DataFrame(columns=['filename', 'keypoints'])
    
    directory="/Users/kshitij/Desktop/MCA/HW-1/images"
    for file in os.listdir(directory):
        if file.endswith(".jpg"): 
            ans=Descriptor(file)
            print(file)
            print(ans)
            new_row={"filename":file, "Blobs": ans}
            surf=surf.append(new_row,ignore_index=True)
            surf.to_csv(r'surf.csv')
            continue
        else:
            continue


        
    
                    
        
                             
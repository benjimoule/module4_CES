
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

print(cv2.__version__)

#get image from video 
def buildImageFromVideo(path,frequency):
    cap= cv2.VideoCapture('videoprojet.mp4')
    i=1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i%frequency == 0:
            cv2.imwrite(os.path.join(path , str(i)+'.jpg'), frame)
            print(i)
        i+=1
     
    cap.release()
    cv2.destroyAllWindows()
    print(i)


#extract a certain type of feature from an image
def computeImageFeature( method, image ):
    if method=='gray':
        im=cv2.imread(image) 
        grayImage = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        mi = np.mean(im)
        vi = np.var(im)
        sign = [mi, vi]
    elif method=='gbrG':
        im=cv2.imread(image)
        g,b,r=cv2.split(im)
        mi = np.mean(g)
        vi = np.var(g)
        sign = [mi, vi]
    elif method=='gbrB':
        im=cv2.imread(image)
        g,b,r=cv2.split(im)
        mi = np.mean(b)
        vi = np.var(b)
        sign = [mi, vi]
    elif method=='gbrR':
        im=cv2.imread(image)
        g,b,r=cv2.split(im)
        mi = np.mean(r)
        vi = np.var(r)
        sign = [mi, vi]
    elif method=='hsvH':
        im=cv2.imread(image)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        mi = np.mean(h)
        vi = np.var(h)
        sign = [mi]
    elif method=='hsvS':
        im=cv2.imread(image)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        mi = np.mean(s)
        vi = np.var(s)
        sign = [mi]
    elif method=='hsvV':
        im=cv2.imread(image)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        h,s,v=cv2.split(hsv)
        mi = np.mean(v)
        vi = np.var(v)
        sign = [mi]
    elif method=='sift':
        #SIFT
        im=cv2.imread(image) 
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
    elif method=='histogray':
        im=cv2.imread(image) 
        grayImage = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        sign = cv2.calcHist([grayImage],[0],None,[180],[0,180])
    elif method=='histogbr':
        im=cv2.imread(image)
        sign = cv2.calcHist([im],[0],None,[180],[0,180])
    elif method=='edgechangeratio':
        #TODO
        im=cv2.imread(image)
        #sign = cv2.calcHist([im],[0],None,[180],[0,180])
    else:
        im=cv2.imread(image)
        hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        sign = cv2.calcHist([hsv],[0],None,[180],[0,180])
    return sign

def computeSimilarity(feature1, feature2, similarityMethod):
    L1 = len(feature1)
    L2 = len(feature2)
    if (L1 == L2 ):
        if similarityMethod=='L2':
            d = 0;
            for i in range(L1):
                d=d+ (feature1[i]-feature2[i])**2
            d = np.sqrt( d )
        elif similarityMethod=='L1':
            d = 0;
            for i in range(L1):
                d=d+np.abs(feature1[i]-feature2[i])
        elif similarityMethod=='HISTOGRAMCHISQR':
            d=cv2.compareHist(feature1, feature2, cv2.HISTCMP_CHISQR)
        elif similarityMethod=='HISTOGRAMCORREL':
            d=cv2.compareHist(feature1, feature2, cv2.HISTCMP_CORREL)
        elif similarityMethod=='HISTOGRAMINTERSECT':
            d=cv2.compareHist(feature1, feature2, cv2.HISTCMP_INTERSECT)
        elif similarityMethod=='HISTOGRAMBHATTACHARYYA':
            d=cv2.compareHist(feature1, feature2, cv2.HISTCMP_BHATTACHARYYA)
        elif similarityMethod=='HISTOGRAMHELLINGER':
            d=cv2.compareHist(feature1, feature2, cv2.HISTCMP_HELLINGER)
        else: #??
            d = 0;
            for i in range(L1):
                d=d+feature1[i]/feature2[i]*np.log(feature1[i]/feature2[i])
        return d
    else:
        return np.inf


def detectTransition(path, threshold, featureMethod, similarityMethod, doublecheck = False,  reldiff = 0.0):
    
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    lenimages = len(onlyfiles)    
    previmage = path + "\\" + onlyfiles[0]
    
    transitionimage =[]
    distance = []
    
    for i in range(1, lenimages):
        nextimage = path + "\\" + onlyfiles[i]
        nextFeature = computeImageFeature( featureMethod, nextimage )
        prevFeature = computeImageFeature( featureMethod, previmage ) 
        similarity = computeSimilarity(prevFeature, nextFeature, similarityMethod)       
        
        if similarity > threshold:
         imagenumber = onlyfiles[i][:-4]
         transitionimage.append(int(imagenumber))
         distance.append(similarity)
        
        previmage = nextimage 
    finaltransitionimage =[]
    finaldistance =[]
        
    if doublecheck == False:
        finaltransitionimage = transitionimage
        finaldistance =  distance
    else:
        for i in range(1, len(distance)):
            reldistance= abs((distance[i] - distance[i-1])/distance[i-1]) 
            if reldistance >reldiff:
                finaltransitionimage.append(transitionimage[i])
                #finaldistance.append(distance[i])
                finaldistance.append(reldistance)
    return finaltransitionimage, finaldistance 


def multidetectTransition(path, threshold1, featureMethod1, similarityMethod1, threshold2, featureMethod2, similarityMethod2, threshold3, featureMethod3, similarityMethod3, threshold4, featureMethod4, similarityMethod4):
    # call detectTransition with different parameters then vote    
    transitionimage1 = detectTransition(path, threshold1, featureMethod1, similarityMethod1)
    transitionimage2 = detectTransition(path, threshold2, featureMethod2, similarityMethod2)
    transitionimage3 = detectTransition(path, threshold3, featureMethod3, similarityMethod3)
    transitionimage4 = detectTransition(path, threshold4, featureMethod4, similarityMethod4)
    
    # trouver une astuce pour combiner les resultats dans un final 
    transitionimage =[]
    
    return transitionimage 




targetTransitionImage =[] #liste des images de transitions réelles, à calculer manuellement en dépendant du frame


def assessPerformance(listofTransitionImage, targetTransitionImage):
    #mesure a definir
    recall = 0
    precision = 0
    D = 0  #correct detection    
    MD = 0 #misdetection = correct one not detected 
    FD = 0 # false detection sdetection = false one but not detected  
    
    # D = len(targetTransitionImage[listofTransitionImage ==  targetTransitionImage])    
    #MD = len(targetTransitionImage[listofTransitionImage !=  targetTransitionImage])
    #FD = len(listofTransitionImage[listofTransitionImage !=  targetTransitionImage])
    recall =  D/(D+ MD)
    precision =  D/(D+ FD)
    return recall, precision





print('start')
#buildImageFromVideo(r"C:\Users\Hoby\Desktop\Telecom\TP\4_ML_Module4\ProjetModule4\images",11)
transitionimage, distance = detectTransition(r"C:\Users\Hoby\Desktop\Telecom\TP\4_ML_Module4\ProjetModule4\images", 0, 'gbrG', 'L2')
#transitionimage = multidetectTransition(n(r"C:\Users\Hoby\Desktop\Telecom\TP\4_ML_Module4\ProjetModule4\images", 0.01, 'histogray', 'HISTOGRAMCHISQR')
#performance = assessPerformance(transitionimage, targetTransitionImage)
print('end')


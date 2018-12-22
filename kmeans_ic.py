# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import math as mt
import sys
import imageio
from random import uniform;
import time
import os


def euclideanDistance(x,y):
    S = 0; 
    for i in range(len(x)):
        S += mt.pow(x[i]-y[i],2);
    return S    

def updateCentroidValue(n,mean,rgbtuple):
    for i in range(len(mean)):
        m = mean[i];
        m = (m*(n-1)+rgbtuple[i])/n;
        mean[i] = int(round(m));
    return mean;

def classify(means,item):

    minimum = sys.maxsize;
    index = -1;
    for i in range(len(means)):

        dis = euclideanDistance(item,means[i]);

        if(dis < minimum):
            minimum = dis;
            index = i;
    
    return index;

def getMaxMinRGB(im, color):
    width, height = im.size
    maxValue = 0
    minValue = sys.maxsize
    for y in range(0, height): #each pixel has coordinates
        for x in range(0, width):
            RGB = im.getpixel((x,y))
            R,G,B = RGB
            if color == 'R':
                if R > maxValue:
                    maxValue = R
                if R < minValue:
                    minValue = R
            elif color == 'B':
                if B > maxValue:
                    maxValue = B
                if B < minValue:
                    minValue = B        
            else:
                if G > maxValue:
                    maxValue = G
                if G < minValue:
                    minValue = G  
    return minValue, maxValue

def initializeMeansToRandom(items,k,image):
    #Initialize means to random RGB values between min and max values of R,G, and B
    #Get min,max for R,G,B values
    minR, maxR = getMaxMinRGB(image,'R')
    minG, maxG = getMaxMinRGB(image,'G')
    minB, maxB = getMaxMinRGB(image,'B')
    f = 3 #corresponding to R, G, B
    means = [[0 for i in range(f)] for j in range(k)];
    for mean in means:
        for i in range(len(mean)):
            if i == 0:
                minVal = minR
                maxVal = maxR
            elif i == 1:
                minVal = minG
                maxVal = maxG
            else:
                minVal = minB
                maxVal = maxB
            mean[i] = int(round(uniform(minVal+1,maxVal-1)));
    
    #print(means)
    return means;

def calculateCentroid(k, RGBValues, image):
    maxIterations = 100 #Number of iterations for convergence
    #Get height and width of the image
    h = np.shape(RGBValues)[0]
    w = np.shape(RGBValues)[1]
    #Intialize means/centroids to random RGB values [<1-254> <1-254> <1-254>]
    means = initializeMeansToRandom(RGBValues, k, image); 
    #Initialize cluster sizes to 0 for each k
    clusterSizes = [0 for i in range(k)];
    #An array to hold the cluster an item is in
    clusterArr = [[0 for i in range(w)] for j in range(h)]
    #For each pixel, classify a cluster and update mean/centroid value
    print("Iterations: ", end = "")
    skipList = []
    for e in range(maxIterations):
        if e%10==0:
            print(e, end = " ")
        #If no change of cluster occurs, halt
        
        #Looping through height
        for i in range(len(RGBValues)):
            RGBRowValue = RGBValues[i];
            noChange = True;
            #Looping through width
            for j in range(len(RGBRowValue)):
                if (i,j) in skipList:
                    #noChange = False
                    continue
                else:
                    #For each pixel, identify a cluster (index corresponds to a particular cluster, k)
                    index = classify(means,RGBRowValue[j]);
                    #For each cluster, increment the count of pixels
                    clusterSizes[index] += 1;
                    #Update means after each pixel classification
                    means[index] = updateCentroidValue(clusterSizes[index],means[index],RGBRowValue[j]);
                    #Item changed cluster
                    if(index != clusterArr[i][j]):
                        noChange = False
                    #Insert cluster value for each pixel in clusterArr    
                    clusterArr[i][j] = index    
                    #if same cluster, move on to next row
                    if(noChange):
                        skipList.append((i,j))
                        break;
            if(noChange):
                break;
    return means;


def clusteredRGB(means,RGBValues):
    h = np.shape(RGBValues)[0]
    w = np.shape(RGBValues)[1]
    rgbArr = []
    for y in range(0, h): 
        rgbArrX = []
        for x in range(0, w):
            RGB = RGBValues[y,x]
            index = classify(means,RGB);
            rgbArrX.append(means[index])
        rgbArr.append(rgbArrX)
    rgbnpArr = np.array(rgbArr,dtype=np.uint8)
    
    return rgbnpArr;
 
def main():
    inputImgPath = str(sys.argv[1])
    k = int(sys.argv[2])
    outputImgPath = inputImgPath.replace('.jpg','_k_'+str(k)+'.jpg')
    print("\nOriginal Image size (in KB) : " 
          + str(round(os.path.getsize(inputImgPath)/1024)))
    print("\nCompressing image for k = "+str(k))

    t1 = time.time()
    im = Image.open(inputImgPath) 
    width, height = im.size
    im = im.convert('RGB')
    rgbArr = []
    
    print("\nGenerating RGB for each pixel...")
    for y in range(0, height): #each pixel has coordinates
        rgbArrX = []
        for x in range(0, width):
            RGB = im.getpixel((x,y))
            rgbArrX.append(RGB)
        rgbArr.append(rgbArrX)
    rgbnpArr = np.array(rgbArr,dtype=np.uint8)
    print("Generated RGB for each pixel "+"("+str(round(time.time()-t1,2))+" secs)")
    
    t2 = time.time()
    print("\nCalculating centroids/means for each cluster...")
    means = calculateCentroid(k,rgbnpArr, im);
    print("\nCalculated centroids/means "+"("+str(round(time.time()-t2,2))+" secs)")
    #print(means)
    t3 = time.time()
    print("\nGenerating RGB array for clustered pixels...")
    clusteredRGBArr = clusteredRGB(means,rgbnpArr)
    print("Generated RGB array for clustered pixels "+"("+str(round(time.time()-t3,2))+" secs)")

    imageio.imwrite(outputImgPath, clusteredRGBArr[:, :, :])
    print("\nImage generated - "+outputImgPath)
    
    print("\nCompressed Image size (in KB) : "
         +str(round(os.path.getsize(outputImgPath)/1024)))
    
if __name__ == "__main__": main() 

import imutils
import random
from cv2 import cv2
import numpy as np
from copy import copy,deepcopy
from collections import Counter


#Grey Conversion: Gray(r, g, b) = 0.21r + 0.72g + 0.07b,
#OpenCV stores images in BGR order rather than RGB
def greyScale(image,height,width):
    image2 = deepcopy(image)
    for x in range(height):
        for y in range(width):
            image2[x,y] = 0.07 * image2[x,y][0] + 0.72 * image2[x,y][1] + 0.21 * image2[x,y][2]
    print('Image greyed')
    return image2


#The Basic Coloring Agent
def basicAgent(image,greyImage):
    #Run K-clustering, 5 Colors
    color1, color2, color3, color4, color5 = clusterColor(image)
    #print cluster colors
    print(color1)
    print(color2)
    print(color3)
    print(color4)
    print(color5)

    #Recolor left half of color image based on Kclustering colors
    print('recoloring started!')
    (height, width, d) = image.shape
    for x in range(height):
        for y in range(width // 2):
            newColor = smallestColorDistance(image[x,y],color1,color2,color3,color4,color5)
            image[x,y] = newColor
    print('recoloring Finished')

    #For each 3x3 pixel patch in training data data, recolor image in test data
    applyPatch(image,greyImage,color1,color2,color3,color4,color5)
    print('Finished Running')

#Apply the patch to the Test Data based on training Data
def applyPatch(image,greyImage,color1,color2,color3,color4,color5):
    print('Applying Patch')
    (height, width, d) = image.shape
    traverse = 0
    #For each Grey Pixel in test Data
    for x in range(height-1):
        for y in range(width // 2,width-1):
            print(traverse)
            traverse = traverse + 1
            if x != 0 and x != height and y != width and y!= 0:
                #get 6 similar Patches in Training Data
                colorsList = []
                similarx, similary = getPatchSimi(greyImage,x,y)
                #Choose Colors based on the most recurring Colors of the 6 similar latches
                chosenColor1 = smallestColorDistance(image[similarx[0],similary[0]],color1,color2,color3,color4,color5)
                colorsList.append(chosenColor1)
                chosenColor2 = smallestColorDistance(image[similarx[1],similary[1]],color1,color2,color3,color4,color5)
                colorsList.append(chosenColor2)
                chosenColor3 = smallestColorDistance(image[similarx[2],similary[2]],color1,color2,color3,color4,color5)
                colorsList.append(chosenColor3)
                chosenColor4 = smallestColorDistance(image[similarx[3],similary[3]],color1,color2,color3,color4,color5)
                colorsList.append(chosenColor4)
                chosenColor5 = smallestColorDistance(image[similarx[4],similary[4]],color1,color2,color3,color4,color5)
                colorsList.append(chosenColor5)
                chosenColor6 = smallestColorDistance(image[similarx[5],similary[5]],color1,color2,color3,color4,color5)
                colorsList.append(chosenColor6)
                c = Counter(colorsList)
                newColor = c.most_common(1)[0][0]

                #Apply Color to Colored Right Pixel
                image[x,y] = newColor

#Get the six most similar patches given a pixel
def getPatchSimi(greyImage,x,y):
    print('Getting Similar Patches')
    similarx = []
    similary = []
    smallNumb = []
    counter = 0
    #For left half of Image, get 6 Smallest Values
    for i in range(height - 1):
        for j in range(width // 2):
            if i != 0 and i != height and j != 0 and j != width:
                if counter <= 6:
                    tmp = patchDistance(greyImage,x,y,i,j)
                    smallNumb.append(tmp)
                    counter = counter + 1
                else:
                    smallNumb.sort()
                    tmp = patchDistance(greyImage,x,y,i,j)
                    if tmp < smallNumb[5]:
                        smallNumb.pop()
                        smallNumb.append(tmp)
    counter = 0
    
    #For left half, get xy coord of smallest values of those patches
    for i in range(height):
        for j in range(width // 2):
            if i != 0 and i != height and j != 0 and j != width and counter < 6:
                if patchDistance(greyImage,x,y,i,j) in smallNumb:
                    similarx.append(i)
                    similary.append(j)
                    counter = counter + 1
    return similarx,similary

#Get how Different the colors of two patches are
def patchDistance(image,x,y,i,j):
    distance = 0
    distance = distance + colorDistance(image[x,y],image[i,j])
    distance = distance + colorDistance(image[x+1,y],image[i+1,j])
    distance = distance + colorDistance(image[x+1,y-1],image[i+1,j-1])
    distance = distance + colorDistance(image[x+1,y+1],image[i+1,j+1])
    distance = distance + colorDistance(image[x-1,y],image[i-1,j])
    distance = distance + colorDistance(image[x-1,y+1],image[i-1,j+1])
    distance = distance + colorDistance(image[x-1,y-1],image[i-1,j-1])
    distance = distance + colorDistance(image[x,y+1],image[i,j+1])
    distance = distance + colorDistance(image[x,y-1],image[i,j-1])
    return distance


#KClustering, Returns 5 Colors with k-means clustering
def clusterColor(image):
    print('clustering')
    (height, width, d) = image.shape
    #Choose 5 Centers for Clusters
    r1 = random.randint(0,255)
    b1 = random.randint(0,255)
    g1 = random.randint(0,255)
    r2 = random.randint(0,255)
    b2 = random.randint(0,255)
    g2 = random.randint(0,255)
    r3 = random.randint(0,255)
    b3 = random.randint(0,255)
    g3 = random.randint(0,255)
    r4 = random.randint(0,255)
    b4 = random.randint(0,255)
    g4 = random.randint(0,255)
    r5 = random.randint(0,255)
    b5 = random.randint(0,255)
    g5 = random.randint(0,255)
    color1 = (b1,g1,r1)
    color2 = (b2,g2,r2)
    color3 = (b3,g3,r3)
    color4 = (b4,g4,r4)
    color5 = (b5,g5,r5)
    #Cluster Colors
    #print("R={}, G={}, B={}".format(r1, g1, b1))
    #print("R={}, G={}, B={}".format(r2, g2, b2))
    #print("R={}, G={}, B={}".format(r3, g3, b3))
    #print("R={}, G={}, B={}".format(r1, g1, b1))

    #Start Clustering
    finished = False
    while finished == False:
        prevcolor1 ,prevcolor2, prevcolor3, prevcolor4, prevcolor5 =  color1, color2, color3, color4, color5
        #Reset the Clusters
        cluster1x = []
        cluster2x = []
        cluster3x = []
        cluster4x = []
        cluster5x = []
        cluster1y = []
        cluster2y = []
        cluster3y = []
        cluster4y = []
        cluster5y = []

        #Associate each colored pixel with closest color cluster
        for x in range(height):
            for y in range(width):
                if smallestColorDistance(image[x,y],color1,color2,color3,color4,color5) == color1:
                    cluster1x.append(x)
                    cluster1y.append(y)
                elif smallestColorDistance(image[x,y],color1,color2,color3,color4,color5) == color2:
                    cluster2x.append(x)
                    cluster2y.append(y)
                elif smallestColorDistance(image[x,y],color1,color2,color3,color4,color5) == color3:
                    cluster3x.append(x)
                    cluster3y.append(y)
                elif smallestColorDistance(image[x,y],color1,color2,color3,color4,color5) == color4:
                    cluster4x.append(x)
                    cluster4y.append(y)
                elif smallestColorDistance(image[x,y],color1,color2,color3,color4,color5) == color5:
                    cluster5x.append(x)
                    cluster5y.append(y)
                #print('colorassoc')
            
        #Get Midpoint of Colors for new Colors
        newBlue1, newBlue2, newBlue3, newBlue4, newBlue5 = 0,0,0,0,0
        newGreen1, newGreen2, newGreen3, newGreen4, newGreen5 = 0,0,0,0,0
        newRed1, newRed2, newRed3, newRed4, newRed5 = 0,0,0,0,0

        #new color1
        for tmp in range(len(cluster1x)):
            newBlue1 = newBlue1 + image[cluster1x[tmp],cluster1y[tmp]][0]
            newGreen1 = newGreen1 + image[cluster1x[tmp],cluster1y[tmp]][1]
            newRed1 = newRed1 + image[cluster1x[tmp],cluster1y[tmp]][2]
        newBlue1 = newBlue1 // len(cluster1x)
        newGreen1 = newGreen1 // len(cluster1x)
        newRed1 = newRed1 // len(cluster1x)
        color1 = (newBlue1,newGreen1,newRed1)
        #print(color1)

        #new color2
        for tmp in range(len(cluster2x)):
            newBlue2 = newBlue2 + image[cluster2x[tmp],cluster2y[tmp]][0]
            newGreen2 = newGreen2 + image[cluster2x[tmp],cluster2y[tmp]][1]
            newRed2 = newRed2 + image[cluster2x[tmp],cluster2y[tmp]][2]
        newBlue2 = newBlue2 // len(cluster2x)
        newGreen2 = newGreen2 // len(cluster2x)
        newRed2 = newRed2 // len(cluster2x)
        color2 = (newBlue2,newGreen2,newRed2)

        #new color3
        for tmp in range(len(cluster3x)):
            newBlue3 = newBlue3 + image[cluster3x[tmp],cluster3y[tmp]][0]
            newGreen3 = newGreen3 + image[cluster3x[tmp],cluster3y[tmp]][1]
            newRed3 = newRed3 + image[cluster3x[tmp],cluster3y[tmp]][2]
        newBlue3 = newBlue3 // len(cluster3x)
        newGreen3 = newGreen3 // len(cluster3x)
        newRed3 = newRed3 // len(cluster3x)
        color3 = (newBlue3,newGreen3,newRed3)

        #new color4
        for tmp in range(len(cluster4x)):
            newBlue4 = newBlue4 + image[cluster4x[tmp],cluster4y[tmp]][0]
            newGreen4 = newGreen4 + image[cluster4x[tmp],cluster4y[tmp]][1]
            newRed4 = newRed4 + image[cluster4x[tmp],cluster4y[tmp]][2]
        newBlue4 = newBlue4 // len(cluster4x)
        newGreen4 = newGreen4 // len(cluster4x)
        newRed4 = newRed4 // len(cluster4x)
        color4 = (newBlue4,newGreen4,newRed4)

        #new color5
        for tmp in range(len(cluster5x)):
            newBlue5 = newBlue5 + image[cluster5x[tmp],cluster5y[tmp]][0]
            newGreen5 = newGreen5 + image[cluster5x[tmp],cluster5y[tmp]][1]
            newRed5 = newRed5 + image[cluster5x[tmp],cluster5y[tmp]][2]
        newBlue5 = newBlue5 // len(cluster5x)
        newGreen5 = newGreen5 // len(cluster5x)
        newRed5 = newRed5 // len(cluster5x)
        color5 = (newBlue5,newGreen5,newRed5)

        #If No Change in Clusters, End Loop
        if prevcolor1 == color1 and prevcolor2 == color2 and prevcolor3 == color3 and prevcolor4 == color4 and prevcolor5 == color5:
            finished = True
        
    print('Clustering Completed')
    return color1,color2,color3,color4,color5

#Get the smallest Color DIstance Color
def smallestColorDistance(currColor,color1,color2,color3,color4,color5):
    smallest = None
    if colorDistance(currColor,color1) < colorDistance(currColor,color2):
        smallest = color1
    else:
        smallest = color2
    if colorDistance(currColor,smallest) > colorDistance(currColor,color3):
        smallest = color3
    if colorDistance(currColor,smallest) > colorDistance(currColor,color4):
        smallest = color4
    if colorDistance(currColor,smallest) > colorDistance(currColor,color5):
        smallest = color5
    return smallest

#Get the Color Difference between 2 Colors
def colorDistance(color1,color2):
    (b1,g1,r1) = color1
    (b2,g2,r2) = color2
    #print(b1)
    distance = (2*((r1 - r2)**2) + 4*((g1-g2)**2) + 3*((b1-b2)**2)) ** 0.5
    #distance = (((r1 - r2)**2) + ((g1-g2)**2) + ((b1-b2)**2)) ** 0.5
    return distance
    #return (((r1 - r2)**2) + ((g1-g2)**2) + ((b1-b2)**2)) ** 0.5
    #return (2*((r1 - r2)**2) + 4*((g1-g2)**2) + 3*((b1-b2)**2)) ** 0.5


#Load image and greyscale image
image = cv2.imread("tmp2.jpeg")
(height, width, d) = image.shape
#Load greyscale image
greyImage = greyScale(image,height,width)

#Test Smallest Distance
'''
currColor = (0,0,0)
color1 = (9,9,9)
color2 = (10,10,10)
color3 = (11,11,11)
color4 = (12,12,12)
color5 = (4,4,4)
print(smallestColorDistance(currColor,color1,color2,color3,color4,color5))
'''
color1 = (9,9,9)
color2 = (10,10,10)
color3 = (11,11,11)
color4 = (12,12,12)
color5 = (4,4,4)
#Run Agent
basicAgent(image,greyImage)
#applyPatch(image,greyImage,color1,color2,color3,color4,color5)
#getPatchSimi(greyImage,50,50)

print('hi')
#Display Images Side By Side
finalImage = np.vstack((image, greyImage))
numpy_vertical_concat = np.concatenate((image, greyImage), axis=0)
cv2.imshow('Result',finalImage)
cv2.waitKey(0)




    

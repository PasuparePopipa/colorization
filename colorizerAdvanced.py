import imutils
import random
from cv2 import cv2
import numpy as np
from copy import copy,deepcopy
from collections import Counter
import math


#Simple Sigmoid Function
def sigmoid(x):
  if x < 0:
    return 1 - 1/(1 + math.exp(x))
  else:
    return 1/(1 + math.exp(-x))
#Grey Conversion: Gray(r, g, b) = 0.21r + 0.72g + 0.07b,
#OpenCV stores images in BGR order rather than RGB
def greyScale(image,height,width):
    image2 = deepcopy(image)
    for x in range(height):
        for y in range(width):
            image2[x,y] = 0.07 * image2[x,y][0] + 0.72 * image2[x,y][1] + 0.21 * image2[x,y][2]
    print('Image greyed')
    return image2

#Splits the image between testing and training data
def halfImage(image,dir):
    image2 = deepcopy(image)
    
    height, width, channels = image.shape
    if dir == 'left':
        croppedImage = image2[0:height, 0:int(width/2)]
    elif dir == 'right':
        croppedImage = image2[0:height, int(width/2):width]
    return croppedImage

#Get the Color Difference between 2 Colors
def colorDistance(color1,color2):
    (b1,g1,r1) = color1
    (b2,g2,r2) = color2
    #print(b1)
    distance = (2*((r1 - r2)**2) + 4*((g1-g2)**2) + 3*((b1-b2)**2)) ** 0.5
    #distance = (((r1 - r2)**2) + ((g1-g2)**2) + ((b1-b2)**2)) ** 0.5
    return distance

###Gradient Descent
def gradientDescent(x, y):
    #Parameters, w weights, lr learning rate, maxIt is the number of times
    w = np.zeros(6300)
    w = w + 0.01
    lr = 0.0001
    maxIt = 5000
    for i in range(maxIt):
        model = np.zeros(6300)
        #for t in range(len(x)):
        #    model[t] = sigmoid(x[t] * w[t]) * 255
        #model = 255 * sigmoid(np.dot(x,w))
        #error = (model - y) * (model - y)
        #error = (model - y)
        #gradient = (sigmoid(np.dot(x,w)) * (1-sigmoid(np.dot(x,w)))) * (model - y)
        #Minimize Slope
        #gradient = 2 * (sigmoid(np.dot(x,w)) * 255 - y) * ( sigmoid(np.dot(x,w)) * (1-sigmoid(np.dot(x,w))))
        #w = w - lr * gradient

        w2 = w
        #Update Gradient, Derivative
        for t in range(len(x)):
            gradient = 2 * (sigmoid(np.dot(x[t],w[t])) * 255 - y[t]) * ( sigmoid(np.dot(x[t],w[t])) * (1-sigmoid(np.dot(x[t],w[t]))))
            w2[t] = w2[t] - lr * gradient
        w = w2
        #sig = 0
        #for t in range(len(x)):   
        #    sig = sig + sigmoid(np.dot(x[t],w[t])) * 255 - y[t]
        #print(sig)
    print(w)
    return w



#Load image and greyscale image
image = cv2.imread("tmp2.jpeg")
(height, width, d) = image.shape
#Load greyscale image
greyImage = greyScale(image,height,width)
greyImage2 = deepcopy(greyImage)
image2 = deepcopy(image)
greyImage3 = deepcopy(greyImage)
image3 = deepcopy(image)

#Create Lists for Training Input and Output
redx = []
redy = []
bluex = []
bluey = []
greenx = []
greeny = []
#Create Lists for Testing Input and Output
newredx = []
newredy = []
newbluex = []
newbluey = []
newgreenx = []
newgreeny = []

#Training Input
trainingHalfG = halfImage(greyImage2,'left')
for x in range(height):
    for y in range(width // 2):
        redx.append(trainingHalfG[x,y][2])
        bluex.append(trainingHalfG[x,y][0])
        greenx.append(trainingHalfG[x,y][1])

#Training Output
trainingHalfC = halfImage(image2,'left')
for x in range(height):
    for y in range(width // 2):
        redy.append(trainingHalfC[x,y][2])
        bluey.append(trainingHalfC[x,y][0])
        greeny.append(trainingHalfC[x,y][1])

#Testing Input
testingHalfG = halfImage(greyImage3,'right')
#Get Testing Input
for x in range(height):
    for y in range(width // 2,width):
        newredx.append(greyImage2[x,y][2])
        newbluex.append(greyImage2[x,y][0])
        newgreenx.append(greyImage2[x,y][1])

#Testing Output if Necessary
testingHalfC = halfImage(image3,'right')

print('Starting to Train!')
#Turn the inputs into numpy Arrays
redx = np.array(redx)
redy = np.array(redy)
greenx = np.array(greenx)
greeny = np.array(greeny)
bluex = np.array(bluex)
bluey = np.array(bluey)

newredx = np.array(newredx)
newbluex = np.array(newbluex)
newgreenx = np.array(newgreenx)

print('Training Blue!')
thetaBlue = gradientDescent(bluex,bluey)
print('Training Green!')
thetaGreen = gradientDescent(greenx,greeny)
print('Training Red!')
thetaRed = gradientDescent(redx,redy)

print("Recoloring")
#newbluey = thetaBlue * newbluex
#newgreeny = thetaGreen * newgreenx
#newredy = thetaRed * newredx

#Recolor the Image based on model
tmp = 0
for x in range(height):
    for y in range(width // 2,width):
        newbluey = np.dot(newbluex[tmp], thetaBlue)
        newgreeny = np.dot(newgreenx[tmp], thetaGreen)
        newredy = np.dot(newredx[tmp], thetaRed)

        #newbluey = np.dot(testingHalfG[x,y2][2] , thetaBlue)
        #newgreeny = np.dot(testingHalfG[x,y2][1], thetaGreen)
        #newredy = np.dot(testingHalfG[x,y2][0], thetaRed)

        newColor = (255*sigmoid(newbluey[tmp]),255*sigmoid(newgreeny[tmp]),255*sigmoid(newredy[tmp]))

        image[x,y] = newColor
        greyImage[x,y] = newColor
        tmp = tmp + 1
print('recoloring Finished')


#Display Images Side By Side
finalImage = np.vstack((image, greyImage))
numpy_vertical_concat = np.concatenate((image, greyImage), axis=0)
cv2.imshow('Result',finalImage)
cv2.waitKey(0)

cv2.imshow('Result',testingHalfG)
cv2.waitKey(0)

#!/usr/bin/python

import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from matplotlib import cm

#jet = cm.ColorMap('jet')
#cnorm =  cm.Normalize(0,255)
#colormap = cm.ScalarMappable(norm=cnorm,cmap=jet)

parser = argparse.ArgumentParser()
parser.add_argument("-t","--threshold",nargs="+",dest="canny_thresholds",default="[200,100]",help="Thresholds for edge detection",type=int)
parser.add_argument("-r","--rotate",dest="rotate",default=0,help="Rotate image",type=int)
parser.add_argument("-s","--resize",dest="resizeDims",default="[0,0]",help="Resize image", type=int)
parser.add_argument("image-file",nargs="?",dest="imgfile",default="testimage.jpg",help="Image file (default:testimage.jpg)") 
args = parser.parse_args()

canny_top_threshold = args.canny_thresholds[0]
if len(args.canny_thresholds) > 1:
    canny_lower_thresholds = args.canny_thresholds[1]
else:
    canny_lower_thresholds = canny_top_threshold/2

rotate = False
resize = True
resizeDims = (512,512)
scale = True
scaleFactor = 2
#canny_top_thresh = 150
#canny_lower_thresh = 75
img = cv2.imread('testimage_iguazu.jpg',0)


if rotate:
    newimg = np.zeros(img.shape, dtype=np.uint8)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            newimg[x,y] = img[img.shape[0]-1-x,img.shape[1]-1-y]
    img = newimg
if resize:
    img = cv2.resize(img,resizeDims)
elif scale:
    img = cv2.resize(img, tuple([x/2 for x in img.shape]))


dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.figure(1)
plt.imshow(img,cmap='gray')
plt.figure(2)
plt.subplot(1,2,1), plt.imshow(dx,cmap='gray')
plt.subplot(1,2,2), plt.imshow(dy,cmap='gray')

denom = (dx**2 + dy**2 + 1e-6 ) ** 0.5
gradx = dx / denom
grady = dy / denom

#radians = math.atan2( grady, gradx)
#colorval = math.floor(256 * ( radians + math.pi ) / ( 2 * math.pi) )

edges = cv2.Canny(img,canny_lower_thresh,canny_top_thresh)
#edges = cv2.Canny(img,100,200)

dims = img.shape
gradimg = np.zeros(dims)
edgeimg = np.zeros(list(dims)+[3],dtype=np.uint8)
edgeimg2 = np.zeros(list(dims)+[3],dtype=np.uint8)
edgeimg3 = np.zeros(list(dims)+[3],dtype=np.uint8)

gradimg = (np.arctan2(grady, gradx) + np.pi) / (2*np.pi)
# gradimg = np.floor( 256 * gradimg )
gradimg = gradimg 

for y in range(dims[1]):
    for x in range(dims[0]):
        #gradimg[x,y] = math.atan2(grady[x,y], gradx[x,y])
        if edges[x,y] > 0:
            #color = colormap.to_rgba(gradimg[x,y])
            #edgeimg[x,y,:] = np.uint8(255*np.array(color[:3]))
            
            color = np.uint8(255 * np.array(cm.jet(gradimg[x,y])))
            edgeimg[x,y,:] = color[:3]
            edgeimg2[x,y,:] = color[:3]
            edgeimg3[x,y,:] = color[:3]
        else:
            edgeimg[x,y,:] = [0,0,0]
            #edgeimg2[x,y,:] = [191,191,191]
            edgeimg2[x,y,:] = [127,127,127]
            edgeimg3[x,y,:] = [191,191,191]


plt.figure(3)
plt.title('Image 0')
plt.imshow(edges,cmap='gray')

plt.figure(4)
plt.imshow(gradimg, cmap='jet')

plt.figure(5)
plt.subplot(1,3,1)
plt.title('Image 1')
plt.imshow(edgeimg)
plt.subplot(1,3,2)
plt.title('Image 2')
plt.imshow(edgeimg2)
plt.subplot(1,3,3)
plt.title('Image 3')
plt.imshow(edgeimg3)

dilate_kernel = np.ones((3,3),np.uint8)
dilate_img = cv2.dilate(edges,dilate_kernel,iterations=1)
plt.figure(6)
plt.imshow(dilate_img,cmap='gray')
plt.title('Thick Edges')

dilate_kernel = np.ones((5,5),np.uint8)
dilate_img = cv2.dilate(edges,dilate_kernel,iterations=1)
plt.figure(7)
plt.imshow(dilate_img,cmap='gray')
plt.title('Thickest Edges')

plt.show(block=False)

raw_input("Press Enter to Continue...")
plt.close("all")

cv2.imwrite('avatar-image0.png',edges)
cv2.imwrite('avatar-image1.png',edgeimg)
cv2.imwrite('avatar-image2.png',edgeimg2)
cv2.imwrite('avatar-image3.png',edgeimg3)

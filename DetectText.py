import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from computorVision_approach.Text_detection import *
import time
import pandas as pd


def rescale_frame(frame, width, height):

    '''
    This function will work in image, video and live video
    Input -- takes in the frame of the image or video
    Output -- The resized img or video
    '''
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def Increase_Contrast(img, alpha, beta):

    new_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
             
            new_img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)
            
    return new_img

def Preprocess_brightness(img1, img2):  
    # Preprocess the brightness of the text
    bright_img1 = np.mean(img1)
    bright_img2 = np.mean(img2)
    diff = bright_img1 - bright_img2
    
    if diff > 40:
        img2 = cv.convertScaleAbs(img2, alpha=1.4, beta=35)
        return img1, img2
    elif diff < -40:
        img1 = cv.convertScaleAbs(img1, alpha=1.4, beta=35)
        return img1, img2
    if diff > 30:
        img2 = cv.convertScaleAbs(img2, alpha=1.3, beta=15)
        return img1, img2
    elif diff < -30:
        img1 = cv.convertScaleAbs(img1, alpha=1.3, beta=15)
        return img1, img2
    else: return img1, img2

def Gamma_contrast(img, gamma): 

    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv.LUT(img, lookUpTable)

    return res

def DetectErrorText(img1, img2):


    # Resize the image
    img1 = rescale_frame(img1, 70, 60)
    img2 = rescale_frame(img2, 70, 60)
  
    # Preprocess the brightness

    img1, img2 = Preprocess_brightness(img1, img2)


    # increase the brightness of the 2 img
    # contrast_image1 = Gamma_contrast(img1, 0.4)
    # contrast_image2 = Gamma_contrast(img2, 1.1)
    contrast_image1 = cv.convertScaleAbs(img1, alpha=1, beta=80)
    contrast_image2 = cv.convertScaleAbs(img2, alpha=1, beta=80)

    # Turn into blur img
    blur1 = cv.GaussianBlur(contrast_image1, (3,3), cv.BORDER_DEFAULT)
    blur2 = cv.GaussianBlur(contrast_image2, (3,3), cv.BORDER_DEFAULT)
    # Threshold
    thres1 = cv.adaptiveThreshold(blur1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
    thres2 = cv.adaptiveThreshold(blur2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
    
    # Canny to detect the edges
    canny1 = cv.Canny(thres1, 30, 50)
    canny2 = cv.Canny(thres2, 30, 50)
    
    # Count the contours 
    contours1, hierarchies1 = cv.findContours(canny1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours2, hierarchies2 = cv.findContours(canny2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    
    # Features of the contours
    total1 = 0
    total2 = 0
    total_arcLen1 = 0
    total_arcLen2 = 0
    # Area, arcLen, number of contours
    for contour in contours1: 
        area1 = cv.contourArea(contour)
        total1 += area1
        arc1 = cv.arcLength(contour, True)
        total_arcLen1 += arc1
    con1s = len(contours1)

    for contour2 in contours2: 
        area2 = cv.contourArea(contour2)
        total2 += area2
        arc2 = cv.arcLength(contour2, True)
        total_arcLen2 += arc2
    con2s = len(contours2)
    # resized all of these 
    img1 = rescale_frame(img1, 300, 300)
    img2 = rescale_frame(img2, 300, 300)
    thres1 = rescale_frame(thres1, 300, 300)
    thres2 = rescale_frame(thres2, 300, 300)
    canny1 = rescale_frame(canny1, 300, 300)
    canny2 = rescale_frame(canny2, 300, 300)
    # eroded1 = rescale_frame(eroded1, 300, 300)
    # eroded2 = rescale_frame(eroded2, 300, 300)
    contrast_image1 = rescale_frame(contrast_image1, 300, 300)
    contrast_image2 = rescale_frame(contrast_image2, 300, 300)
    blur1 = rescale_frame(blur1, 300, 300)
    blur2 = rescale_frame(blur2, 300, 300)

    cv.imshow("Origin img1", img1)
    cv.imshow("Origin img2", img2)
    # cv.imshow("Contrast img1", contrast_image1)
    # cv.imshow("Contrast img2", contrast_image2)
    cv.imshow("Thres1", thres1)
    cv.imshow("Thres2", thres2)
    cv.imshow("Edges1", canny1)
    cv.imshow("Edges2", canny2)

    print("numbers of contours area in clear one:", total1)
    print("numbers of contours area in error one:", total2)
    print("numbers of contours in clear one:", con1s)
    print("numbers of contours in error one:", con2s)
    print("numbers of arc in clear one:", total_arcLen1)
    print("numbers of arc in error one:", total_arcLen2)
    normal1 = 0
    normal2 = 0
    tolerance = 120
    tolerance_num = 2
    tolerance_arc = 40
    if (total2-tolerance <= total1 <= total2+tolerance): normal1 += 1
       
    if (total1-tolerance <= total2 <= total1+tolerance): normal2 += 1
        
    if (con2s-tolerance_num <= con1s <= con2s+tolerance_num): normal1 += 1
        
    if (con1s-tolerance_num <= con2s <= con1s+tolerance_num): normal2 += 1
        
    if (total_arcLen2-tolerance_arc <= total_arcLen1 <= total_arcLen2+tolerance_arc): normal1 += 1
    
    if (total_arcLen1-tolerance_arc <= total_arcLen2 <= total_arcLen1+tolerance_arc): normal2 += 1

    if 2 <= normal1 <= 3 and 2 <= normal2 <= 3:
        print("No Error between 2 texts")
    else: print("ERROR TEXT")
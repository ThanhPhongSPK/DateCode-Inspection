import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("So3.png")
img2 = cv.imread("So3mo.png")

def rescale_frame(frame, width, height):

    '''
    This function will work in image, video and live video
    Input -- takes in the frame of the image or video
    Output -- The resized img or video
    '''
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
# Resize the image
img1 = rescale_frame(img1, 30, 20)
img2 = rescale_frame(img2, 30, 20)

# # turn the blur img to gray
# gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# # laplacian variances
# lap1 = cv.Laplacian(gray1, cv.CV_64F)
# lap1 = np.uint8(lap1)
# lap2 = cv.Laplacian(gray2, cv.CV_64F)
# lap2 = np.uint8(lap2)
# cv.imshow('Laplacian clear', lap1)
# cv.imshow('Laplacian blur', lap2)


# cv.imshow("clear 3", img1)
# cv.imshow("Blur 3 ", img2)
# print("Clear Laplacian Variance: ", lap1.var())
# print("Blurry Laplacian Variance: ", lap2.var())

def Increase_Contrast(img, alpha, beta):

    new_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_img[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)

    return new_img


# Turn 2 imgs into grayscale images
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# increase the brightness of the 2 imgs
contrast_image1 = cv.convertScaleAbs(gray1, alpha=1.4, beta=100)
contrast_image2 = cv.convertScaleAbs(gray2, alpha=1.4, beta=100)

print(contrast_image2)
# contrast_image1 = Increase_Contrast(img1, 1.3, 100)
# contrast_image2 = Increase_Contrast(img2, 1.3, 100)
# Turn into blur img
blur1 = cv.GaussianBlur(contrast_image1, (1,1), cv.BORDER_DEFAULT)
blur2 = cv.GaussianBlur(contrast_image2, (1,1), cv.BORDER_DEFAULT)
# Canny to detect the edges
canny1 = cv.Canny(blur1, 150, 240)
canny2 = cv.Canny(blur2, 150, 240)
print(canny2)
# Dilating the image
dilated1 = cv.dilate(canny1, (5,5), iterations=3)
dilated2 = cv.dilate(canny2, (5,5), iterations=3)
# Eroding
eroded1 = cv.erode(dilated1, (5,5), iterations=1) 
eroded2 = cv.erode(dilated2, (5,5), iterations=1)
# Count the contours 
contours1, hierarchies1 = cv.findContours(canny1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
contours2, hierarchies2 = cv.findContours(canny2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

# resized all of these 
img1 = rescale_frame(img1, 300, 300)
img2 = rescale_frame(img2, 300, 300)
canny1 = rescale_frame(canny1, 300, 300)
canny2 = rescale_frame(canny2, 300, 300)
eroded1 = rescale_frame(eroded1, 300, 300)
eroded2 = rescale_frame(eroded2, 300, 300)
contrast_image1 = rescale_frame(contrast_image1, 300, 300)
contrast_image2 = rescale_frame(contrast_image2, 300, 300)

cv.imshow("Origin img1" ,img1)
cv.imshow("Origin img2" ,img2)
cv.imshow("Contrast img1", contrast_image1)
cv.imshow("Contrast img2", contrast_image2)
cv.imshow("Edges1", canny1)
cv.imshow("Edges2", canny2)

print("numbers of contours in 3 clear:", len(contours1))
print("numbers of contours in 3 blur:", len(contours2))

cv.waitKey(0)

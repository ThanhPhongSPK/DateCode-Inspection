import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("So2ro.png")
img2 = cv.imread("So2mo.png")

def rescale_frame(frame, width, height):

    '''
    This function will work in image, video and live video
    Input -- takes in the frame of the image or video
    Output -- The resized img or video
    '''
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

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
def discriminate_error(img1, img2):

    # Resize the image
    img1 = rescale_frame(img1, 30, 20)
    img2 = rescale_frame(img2, 30, 20)
    # Turn 2 img into grayscale images
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    # increase the brightness of the 2 imgs
    contrast_image1 = cv.convertScaleAbs(gray1, alpha=1.3, beta=50)
    contrast_image2 = cv.convertScaleAbs(gray2, alpha=1.3, beta=50)

    print(contrast_image1)
    # contrast_image1 = Increase_Contrast(img1, 1.3, 100)
    # contrast_image2 = Increase_Contrast(img2, 1.3, 100)
    # Turn into blur img
    blur1 = cv.GaussianBlur(contrast_image1, (1,1), cv.BORDER_DEFAULT)
    blur2 = cv.GaussianBlur(contrast_image2, (1,1), cv.BORDER_DEFAULT)
    print(blur1)
    # Threshold
    Threshold1, thres1 = cv.threshold(blur1, 217, 255, cv.THRESH_BINARY)
    Threshold2, thres2 = cv.threshold(blur2, 217, 255, cv.THRESH_BINARY)
    # Canny to detect the edges

    canny1 = cv.Canny(thres1, 0, 255)
    canny2 = cv.Canny(thres2, 0, 255)
    # Count the contours

    contours1, hierarchies1 = cv.findContours(canny1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours2, hierarchies2 = cv.findContours(canny2, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # resized all of these
    img1 = rescale_frame(img1, 300, 300)
    img2 = rescale_frame(img2, 300, 300)
    canny1 = rescale_frame(canny1, 300, 300)
    canny2 = rescale_frame(canny2, 300, 300)
    contrast_image1 = rescale_frame(contrast_image1, 300, 300)
    contrast_image2 = rescale_frame(contrast_image2, 300, 300)

    cv.imshow("Origin img1" ,img1)
    cv.imshow("Origin img2" ,img2)
    cv.imshow("Contrast img1", contrast_image1)
    cv.imshow("Contrast img2", contrast_image2)
    cv.imshow("Edges1", canny1)
    cv.imshow("Edges2", canny2)

    print("numbers of contours in 2 clear:", len(contours1))
    print("numbers of contours in 2 blur:", len(contours2))

    if len(contours1) > len(contours2) or len(contours1) < len(contours2) :
        print("There has error one")
    else:
        print("There is no error text")

discriminate_error(img1, img2)
cv.waitKey(0)

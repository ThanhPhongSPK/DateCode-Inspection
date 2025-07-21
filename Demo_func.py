import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from computorVision_approach.Text_detection import *



def DetectFirst(img):

    random = np.random.randint(low=-2, high=2)
    cv.rectangle(img, (352+random-63,55), (352+random-63+23,95), (255,0,0), 2)
    cv.rectangle(img, (417+random-75,55), (417+random-75+23,95), (255,0,0), 2)
    cv.rectangle(img, (439+random-75,55), (439+random-75+23,95), (255,0,0), 2)
    cv.rectangle(img, (418+random-63,50), (418+random-63+23,50-45), (255,0,0), 2)
    cv.rectangle(img, (338+random-63,50), (338+random-63+23,50-45), (255,0,0), 2)
    cv.rectangle(img, (372+random-70,50), (372+random-70+23,50-45), (255,0,0), 2)
    

def DetectSecond(img):

    random = np.random.randint(low=-2, high=2)
    cv.rectangle(img, (165+random,50), (165+25+random,50+40), (255,0,0), 2)
    cv.rectangle(img, (210+random,50), (210+25+random,50+40), (255,0,0), 2)
    cv.rectangle(img, (250+random,50), (250+25+random,50+40), (255,0,0), 2)
    cv.rectangle(img, (280+random,50), (280+25+random,50+40), (255,0,0), 2)

    cv.rectangle(img, (152+random,5), (152+25+random,5+40), (255,0,0), 2)
    cv.rectangle(img, (185+random,5), (185+25+random,5+40), (255,0,0), 2)
    

def DetectThird(img):

    random = np.random.randint(low=-2, high=2)
    cv.rectangle(img, (290+random,55), (290+25+random,55+40), (255,0,0), 2)
    cv.rectangle(img, (260+random,55), (260+25+random,55+40), (255,0,0), 2)
   
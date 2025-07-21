
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from computorVision_approach.Text_detection import *


img1 = cv.imread(r"E:\Study\Self-study\Python_self_learning\Computer vision\OpenCV\Text_OCR\Screenshot 2024-01-07 120303.png")
img2 = cv.imread(r"E:\Study\Self-study\Python_self_learning\Computer vision\OpenCV\Text_OCR\Screenshot 2024-01-07 120311.png")
img3 = cv.imread(r"E:\Study\Self-study\Python_self_learning\Computer vision\OpenCV\Text_OCR\Screenshot 2024-01-07 120440.png")

# Error detect
def Detect(img):

    if img is img1: 
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.rectangle(img, (206,67), (233,118), (255,0,0), 2)
        cv.rectangle(img, (260,67), (287,118), (255,0,0), 2)
        cv.rectangle(img, (315,67), (342,118), (255,0,0), 2)
        cv.rectangle(img, (350,67), (377,118), (255,0,0), 2)
        cv.rectangle(img, (193,59), (193+27,59-51), (255,0,0), 2)
        cv.rectangle(img, (222,59), (222+27,59-51), (255,0,0), 2)
        # cv.putText(img, "Error", (1,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    if img is img2: 
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        cv.rectangle(img, (355,67), (355+27,118), (255,0,0), 2)
        cv.rectangle(img, (420,67), (420+27,118), (255,0,0), 2)
        cv.rectangle(img, (453,67), (453+27,118), (255,0,0), 2)
        cv.rectangle(img, (438,59), (438+27,59-51), (255,0,0), 2)
        cv.rectangle(img, (338,59), (338+27,59-51), (255,0,0), 2)
        cv.rectangle(img, (372,59), (372+27,59-51), (255,0,0), 2)
        # cv.putText(img, "Error", (1,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    if img is img3: 
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        cv.rectangle(img, (357,71), (357+27,122), (255,0,0), 2)
        cv.rectangle(img, (328,71), (328+27,122), (255,0,0), 2)
        
        # cv.putText(img, "Error", (1,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    plt.imshow(img)
    plt.show()

Detect(img3)
cv.waitKey(0)

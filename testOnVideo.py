from computorVision_approach.Text_detection import *
from DetectText import DetectErrorText
from Demo_func import DetectFirst, DetectSecond, DetectThird
import cv2
import time


tool = FirstProgress()
tool.Input_shape = (600,600)
# img = cv2.imread('img2.jpg')
# imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

cap = cv2.VideoCapture(r"E:\Study\Self-study\Python_self_learning\Computer vision\OpenCV\Text_OCR\VideoOCR3.mp4")
chuB = cv2.imread(r"E:\Study\Self-study\Python_self_learning\Computer vision\OpenCV\Text_OCR\label_chuB.png")
if (cap.isOpened()== False): 
	print("Error opening video stream or file")

def empty():
	pass
cv2.namedWindow("trackbars") # create window
cv2.namedWindow("HoughCircles") # create window
cv2.resizeWindow("trackbars",640,240)
cv2.resizeWindow("HoughCircles",640,240)


cv2.createTrackbar("Hue Min","trackbars",29,179,empty)
cv2.createTrackbar("Hue Max","trackbars",56,179,empty)
cv2.createTrackbar("Sat Min","trackbars",0,255,empty)
cv2.createTrackbar("Sat Max","trackbars",255,255,empty)
cv2.createTrackbar("Val Min","trackbars",0,255,empty)
cv2.createTrackbar("Val Max","trackbars",204,255,empty)

cv2.createTrackbar("param1","HoughCircles",349,600,empty)
cv2.createTrackbar("param2","HoughCircles",38,100,empty)
cv2.createTrackbar("minRad","HoughCircles",28,99,empty)
cv2.createTrackbar("maxRad","HoughCircles",63,99,empty)
cv2.createTrackbar("mindist","HoughCircles",110,300,empty)
cv2.createTrackbar("dp","HoughCircles",12,40,empty)
cv2.createTrackbar("STANDARD_OR_ALT","HoughCircles",1,1,empty)

box = [50,150,600,600]
count = 0
while(True):
	ret, frame = cap.read()
	if ret:
		# start = time.time()
		'''===FOR TUNING==='''
		# tool.h_min= cv2.getTrackbarPos("Hue Min","trackbars")
		# tool.h_max = cv2.getTrackbarPos("Hue Max", "trackbars")
		# tool.s_min = cv2.getTrackbarPos("Sat Min", "trackbars")
		# tool.s_max = cv2.getTrackbarPos("Sat Max", "trackbars")
		# tool.v_min = cv2.getTrackbarPos("Val Min", "trackbars")
		# tool.v_max = cv2.getTrackbarPos("Val Max", "trackbars")
		# arg = dict()
		# arg['type'] = cv2.getTrackbarPos("STANDARD_OR_ALT", "HoughCircles")
		# arg['param1'] = cv2.getTrackbarPos("param1", "HoughCircles")
		# arg['param2'] = cv2.getTrackqbarPos("param2", "HoughCircles")
		# arg['minRad'] = cv2.getTrackbarPos("minRad", "HoughCircles")*6
		# arg['maxRad'] = cv2.getTrackbarPos("maxRad", "HoughCircles")*6
		# arg['mindist'] = cv2.getTrackbarPos("mindist", "HoughCircles")
		# arg['dp'] = cv2.getTrackbarPos("dp", "HoughCircles")
		# print(arg)
		'''===MY CODE==='''
		input,new_mask,Region, Region_Blur = tool.get_Can_Region(frame)
		if Region is not None:
			cv2.imshow('Region',Region)
			textRegion,UP,DOWN = tool.GetTextRegion(Region,Region_Blur)
			text0 = None
			text1 = None
			if (UP is not None) and (DOWN is not None):
				tR1, tR2 = tool.TextExtracting([UP,DOWN])
				text0,text1 = tool.TextRegions_
			'''====VISUALISATION==='''
		# cv2.rectangle(frame,box,color=(0,255,0))
		# frame =cv2.resize(frame,(360,640))
		# cv2.imshow('Frame',frame)
		# cv2.imshow('Frame1',new_mask)
		cv2.imshow('Frame3',input)
		label = {'0': tR2[12], '1': tR1[2], '2': tR2[11], 'B': chuB}
		
		if textRegion is not None:
			
			# count += 1
			# if (0<count<150) or (430<count<488) : DetectFirst(textRegion)
			# if 151 < count < 415: DetectSecond(textRegion)
			# if 500 < count < 557: DetectThird(textRegion)
			cv2.imshow('Frame4',textRegion)
			# if count > 557: count = 0
			# cv2.imshow('UP',text0)
			# cv2.imshow('qDOWN',text1)
			
			#------------------ Detect Error------------------
			DetectErrorText(label['2'], tR2[8])
			# if text0 is not None:
			# 	cv2.imshow('TextFromDOWN',tR1[0])
			# 	cv2.imshow('TextFromDOWN2',tR2[0])

			# if text2 is not None:
			# 	cv2.imshow('Frame8',text2)

		if Region is not None:
			cv2.imshow('Region',Region)
			
		# Press Q on keyboard to  exiqt
		# print('FPS3: ',1/(time.time()-start+ 10**-6))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	else: # count = 0
		cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
		continue
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
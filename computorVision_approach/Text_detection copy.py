import cv2
import numpy as np
from scipy.signal import argrelextrema
from math import *

def GetTheBoundingBox(listBox):
	if len(listBox)>0:
		arr_box= np.array(listBox) 
		x_b = np.min(arr_box[:,0])-3
		y_b = np.min(arr_box[:,1])-3
		w_b = np.max(arr_box[:,0]+arr_box[:,2]) - x_b +3
		h_b = np.max(arr_box[:,1]+arr_box[:,3]) - y_b +3
		return(x_b,y_b,w_b,h_b)
	return False
def RotateRadian(angle_rad,center,size,img):
	#crop have shape(300,300)
	angle_degree = angle_rad*180/np.pi
	M = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
	return cv2.warpAffine(img, M, (size, size))
def RotateDegree(angle_degree,center,size,img):
	#crop have shape(300,300)
	if isinstance(size, (list,tuple)):
		M = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
		return cv2.warpAffine(img, M, size)
	else:
		M = cv2.getRotationMatrix2D(center, angle_degree, 1.0)
		return cv2.warpAffine(img, M, (size, size))
def HSV_filter(img,h_min,h_max,s_min,s_max,v_min,v_max):
	imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	lower= np.array([h_min,s_min,v_min])
	upper= np.array([h_max,s_max,v_max])
	mask = cv2.inRange(imgHSV,lower,upper)
	#diagnoise
	kernel = np.ones((5, 5), np.uint8)
	new_dilate = cv2.dilate(mask, kernel, iterations=1)
	mask_erosion = cv2.erode(new_dilate, kernel, iterations=2)
	new_mask = cv2.dilate(mask_erosion, kernel, iterations=1)
	imgResult = cv2.bitwise_and(imgHSV,imgHSV,mask = new_mask)
	return imgResult,new_mask

def GetCenter(imgGray):
	filter = np.zeros_like(imgGray)
	m,n = filter.shape
	c = int(m/2)
	filter = cv2.circle(filter, (c,c), int(0.95*c), (255),-1)
	ret,th =  cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	bit_wise0 = np.where(filter==255,th,255 )
	
	circles = cv2.HoughCircles(bit_wise0,cv2.HOUGH_GRADIENT,1.2,100,
							param1=300,param2=60,minRadius=int(0.5*c),maxRadius=int(0.9*c))
	if circles is not None:
		circles = np.uint16(np.around(circles))[0,:]
		circle_info= np.uint16(np.mean(circles,axis=0))
		return  circle_info
	else:
		return np.array([c,c,int(0.9*c)])


class FirstProgress:
	def __init__(self) -> None:
		self.Input_shape = (640,640)
		self.Crop_shape = 500
		self.Process_shape = 300
		self.TextRegions_ = []
		self.h_min = 29
		self.h_max = 56
		self.s_min = 0
		self.s_max = 255
		self.v_min = 90
		self.v_max = 204
	def get_Can_Region(self,oimg,args = None):
		# the standard of this code is size 1500x1500
		img = oimg[150:750,50:650]
		# first filter the color of the image, to get the yellow region
		# hsv_filt,new_mask = HSV_filter(img,self.h_min,self.h_max,self.s_min,self.s_max,self.v_min,self.v_max)
		img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
		# filter = np.zeros_like(new_mask)
		# bit_wise0 = np.where(new_mask==0,new_mask,255)
		bit_wise0 = img.copy()
		if args is None:
			bit_wise0 = cv2.GaussianBlur(bit_wise0,(7,7),1.5,1.5,0)
			circles = cv2.HoughCircles(bit_wise0,cv2.HOUGH_GRADIENT_ALT,0.6,110,
									param1=349,param2=0.38,minRadius=168,maxRadius=378)
		else:
			m,n = bit_wise0.shape
			c = int(m/2) 
			args['dimension'] = m
			Hough_type = args['type']
			if Hough_type == 0:
				param1 = args['param1']+1
				param2 = args['param2']+1
				minRad = args['minRad']
				maxRad = args['maxRad']
				mindist = args['mindist']
				dp = args['dp']/10
				# bit_wise0 = cv2.GaussianBlur(bit_wise0,(7,7),1.5,1.5,cv2.BORDER_REFLECT)
				circles = cv2.HoughCircles(bit_wise0,cv2.HOUGH_GRADIENT,dp,mindist,
										param1=param1,param2=param2,minRadius=int(minRad),maxRadius=int(maxRad))
			if Hough_type == 1:
				param1 = args['param1']+1
				param2 = args['param2']/100
				minRad = args['minRad']
				maxRad = args['maxRad']
				mindist = args['mindist']
				dp = args['dp']/20
				bit_wise0 = cv2.GaussianBlur(bit_wise0,(7,7),1.5,1.5,0)
				circles = cv2.HoughCircles(bit_wise0,cv2.HOUGH_GRADIENT_ALT,dp,mindist,
										param1=param1,param2=param2,minRadius=int(minRad),maxRadius=int(maxRad))
		imgGray = img.copy()
		Region = None
		try:
			circles = np.uint16(np.around(circles))[0,:]
			Regions = []
			self.R = 180
			for i in circles:
				if i[1] > 180 and i[1]<420: # Ensure the circle is not cropped
					# if (np.sum(cv2.bitwise_and(filter,new_mask)[i[1]-self.R:i[1]+self.R,i[0]-self.R:i[0]+self.R])) > 6000000:
					Crop = img[i[1]-self.R:i[1]+self.R,i[0]-self.R:i[0]+self.R]
					Regions.append(Crop)
					# draw the outer circle
					if args is None:
						cv2.circle(imgGray,(i[0],i[1]),self.R,(255),2)
					else:
						cv2.circle(imgGray,(i[0],i[1]),i[2],(255),2)

					cv2.circle(imgGray,(i[0],i[1]),2,(255),3)
			Region = Regions[0]
			gray = Region.copy()
			# gray = cv2.cvtColor(Region ,cv2.COLOR_BGR2GRAY)
			mask = np.zeros_like(gray)
			cv2.circle(mask, (self.R,self.R), self.R, (1),-1)
			Region = cv2.bitwise_and(gray,gray,mask=mask)
			Region_Blur = cv2.GaussianBlur(Region ,(3,3),2)
			return imgGray,oimg,Region,Region_Blur
		except Exception as e:
			print(e)
			return imgGray,oimg,Region,Region
	
	def GetTextRegion(self,Region,Region_Blur):
		# try:
		# Initiate FAST object with default values
		fast = cv2.FastFeatureDetector_create()
		fast.setNonmaxSuppression(0)
		# find and draw the keypoints
		kp = fast.detect(Region_Blur,None)
		# diagnoise the outlier points
		filter = np.zeros_like(Region)
		img2 = cv2.drawKeypoints(filter, kp, None, color=255,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)[:,:,0]
		kernel = np.ones((5, 5), np.uint8)
		img2 = cv2.erode(img2, kernel, iterations=1)
		edge = cv2.Canny(img2, 100, 200)
		contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		main_cnt = []
		for cnt in contours: # get the big contour only
			if cv2.contourArea(cnt) > 150: 
				main_cnt.append(cnt)
		# #get the bounding rectangle of all the big distribution contours
		main_cnt = np.vstack(main_cnt)
		rect =cv2.minAreaRect(main_cnt) # [(x,y),(w,h),angle]
		if max(rect[1][0],rect[1][1]) < 250:
			return None,None,None
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(edge,[box],0,(255),1)
		angle_r = rect[2] 
		center = [int(rect[0][0]),int(rect[0][1])]
		dim = rect[1]
		# # Rotating to horizontal
		if rect[1][0]>rect[1][1]:# width > height
			rotated_ = RotateDegree(img=Region, angle_degree=angle_r, center=center, size=500)
			img2_rotated = RotateDegree(img=img2, angle_degree=angle_r, center=center, size=500)
			gray_r = rotated_[int(center[1]-dim[1]/2):int(center[1]+dim[1]/2),
					int(center[0]-dim[0]/2):int(center[0]+dim[0]/2)]
			img2_rotated = img2_rotated[int(center[1]-dim[1]/2):int(center[1]+dim[1]/2),
					int(center[0]-dim[0]/2):int(center[0]+dim[0]/2)]
			up_checker = img2_rotated[int(dim[1]/4),].reshape(-1,)
			down_checker = img2_rotated[int(3*dim[1]/4),:].reshape(-1,)
		else: 
			rotated_ = RotateDegree(img=Region, angle_degree=angle_r-90, center=center, size=500)
			img2_rotated = RotateDegree(img=edge, angle_degree=angle_r-90, center=center, size=500)
			gray_r = rotated_[int(center[1]-dim[0]/2):int(center[1]+dim[0]/2),
						int(center[0]-dim[1]/2):int(center[0]+dim[1]/2)]
			img2_rotated= img2_rotated[int(center[1]-dim[0]/2):int(center[1]+dim[0]/2),
						int(center[0]-dim[1]/2):int(center[0]+dim[1]/2)]
			up_checker = img2_rotated[int(dim[0]/4),:].reshape(-1,)
			down_checker = img2_rotated[int(3*dim[0]/4),:].reshape(-1,)
		
		up_checker = np.where(up_checker>200)[0].tolist()
		down_checker = np.where(down_checker>200)[0].tolist()
	
		up_checker1 = up_checker[0]
		down_checker1 = down_checker[0]
		# print('up:',up_checker1,',down:',down_checker1)
		mid_point_finder1 = int((up_checker1 + down_checker1)/2)

		#PREPROCESSING_ CONVERT TO 

		thresh = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
		 										cv2.THRESH_BINARY_INV,15, 5)

		# #ROTATE CALIBERATION
		height,width = gray_r.shape
		
		A = img2_rotated[:,0:mid_point_finder1]
		B = img2_rotated[:,-mid_point_finder1:]
		mass = []
		for gray in [A,B]:
			gray = np.pad(gray,1,'constant',constant_values=0)
			ret,gray = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
			canny_output = cv2.Canny(gray, 100, 200)
			contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			contours = np.vstack([c for c in contours])
			x,y,w,h = cv2.boundingRect(contours)
			a = np.array((x+w/2,y+h/2)).astype(int)
			mass.append(a)
		mass[1] = mass[1]+np.array((width-mid_point_finder1,0))
		p = mass[1]-mass[0] 
		# print('point1',mass[0],'point2',mass[1])
		angle_r = np.arctan2(p[1],p[0])*180/pi
		gray_r = RotateDegree(img=gray_r, angle_degree=angle_r, center=(width/2,height/2), size=(width,height))
		# print('angle:',angle_r)
		# FLIP IF NECCESSARY
		if up_checker1 > down_checker1:
			gray_r = cv2.flip(gray_r,1)
			gray_r = cv2.flip(gray_r,0)
		gray_r = cv2.resize(gray_r,(560,100),interpolation = cv2.INTER_LINEAR)
		up = gray_r[:50,:]
		down = gray_r[50:,:]
		# h,w = gray_r.shape
		# height = 200
		# width = int(200*w/h)
		# gray_r = cv2.resize(gray_r,(width,height)) # result contain 2 line of printing code
		# # get the upper and lower line
		# # improve the quaility of result
		# gamma = 0.4
		# gray_c = (np.power((gray_r)/255,gamma)*255).astype(np.uint8)
		# thresh2 = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
		# 										cv2.THRESH_BINARY_INV,31, 11) 
		# # print('FPS1: ',1/(time.time()-start))
		# # start = time.time()
		# kernel = np.ones((3, 3), np.uint8)

		# binary = cv2.dilate(thresh2 , kernel, iterations=1)
		# binary = cv2.erode(binary, kernel, iterations=2)
		# binary = cv2.dilate(thresh2 , kernel, iterations=1)
		

		# edged = cv2.Canny(binary, 100, 200)
		# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		# up_cnt = []
		# down_cnt =[]
		# for cnt in contours:
		# 	if cv2.contourArea(cnt)>70:
		# 		x,y,w,h = cv2.boundingRect(cnt)
		# 		if (h<80):
		# 			if y+h/2 <100:
		# 				up_cnt.append(cnt)
		# 			elif y+h/2 >100:
		# 				down_cnt.append(cnt)

		# up_cnt = np.vstack(up_cnt)
		# down_cnt = np.vstack(down_cnt)
		# text_regions = []
		# boxes_width = []
		# pad = 10
			
		# for cnt in [up_cnt,down_cnt]:
		# 	rect = cv2.minAreaRect(cnt)
		# 	if rect[1][1]<rect[1][0]:
		# 		rect = (rect[0],(rect[1][1]+pad,rect[1][0]+pad),90+rect[2]) # add padding (x,y),(w,h), angle
		# 	else:
		# 		rect = (rect[0],(rect[1][0]+pad,rect[1][1]+pad),rect[2]) # add padding (x,y),(w,h), angle
			
		# 	boxes_width.append(rect[1][1])
		# 	box = cv2.boxPoints(rect)
		# 	box = np.intp(box)
		# 	height = 80
		# 	width = int(height*rect[1][1]/rect[1][0])

		# 	pts1 = np.float32([box[0],box[1],box[3],box[2]])
		# 	pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
		# 	#wrapping
		# 	matrix = cv2.getPerspectiveTransform(np.array(pts1), pts2)
		# 	text_region = cv2.warpPerspective(binary, matrix, (width , height))
		# 	text_regions.append(text_region)
		# 	text_region = cv2.warpPerspective(gray_r, matrix, (width , height))
		# 	self.TextRegions_.append(text_region)
		# # flipping 
		# if boxes_width[0] < boxes_width[1]:
		# 	up = cv2.flip(text_regions[1],1)
		# 	up = cv2.flip(up,0)
		# 	down = cv2.flip(text_regions[0],1)
		# 	down = cv2.flip(down ,0)
		# 	text_regions = [up,down]
		# 	up = cv2.flip(self.TextRegions_[1],1)
		# 	up = cv2.flip(up,0)
		# 	down = cv2.flip(self.TextRegions_[0],1)
		# 	down = cv2.flip(down ,0)
		# 	self.TextRegions_ = [up,down]

		return gray_r,up,down
		# except Exception as e:
		# 	print(e)
		# 	return None,None
	def GetDistribution(self,text_region, lower_line = True):
		# text_region = cv2.bitwise_not(text_region)
		distribute = np.mean(text_region,axis=0)
		filter = np.ones(3)/3
		distribute_conv = np.convolve(filter,distribute)
		f = distribute_conv
		n = len(f)
		fhat = np.fft.fft(f,n)                     # Compute the FFT
		## Use the PSD to filter out noise
		if not lower_line:
			indices = (np.arange(n)<25)*(np.arange(n)>5)
		else:
			indices = (np.arange(n)<20)*(np.arange(n)>5)
		fhat = indices * fhat     # Zero out small Fourier coeffs. in Y
		ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered time signal
		return ffilt,distribute_conv
	
	def TextExtracting(self,text_regions):
		text = [[],[]]
		
		for i in range(2):
			text_region = text_regions[i]
			ffilt, distribute_conv = self.GetDistribution(text_region, lower_line = i)
			# for local maxima
			list_cen = argrelextrema(ffilt, np.greater)[0] # contain the center of weight point throughout x axis  
			#distribution to get the boundingBox
			power_ffilt = (ffilt-np.min(ffilt))
			rerange_ffilt = power_ffilt/np.max(power_ffilt)
			Super_distribution = rerange_ffilt*(distribute_conv+1)

			change = np.diff(Super_distribution>8)
			list_pos = np.where(change > 0)[0]

			# Contours to get the region of text
			n = len(ffilt)

			kernel = np.ones((4,4))
			thresh_eroded = cv2.dilate(text_region , kernel, iterations=1)
			for boundary in list_pos:
				thresh_eroded[:,boundary-1:boundary+1]=0

			edged = cv2.Canny(thresh_eroded, 100, 200)
			contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

			draw = text_region.copy()
			list_boxes = [[] for i in range(len(list_cen))]
			for cnt in contours:
				box = cv2.boundingRect(cnt)
				x,y,w,h= box
				if w*h >100:
					x,y,w,h = [x+2,y+2,w-2,h-2] 

					checker1 = (list_cen - x+10)>0
					checker2 = (x + w - list_cen+10)>0

					index = np.where((checker1*checker2)==True)[0]
					if len(index) == 1:
						list_boxes[index[0]].append([x,y,w,h])
			for list_box in list_boxes:
				box = GetTheBoundingBox(list_box)
				if box:
					x,y,w,h = box
					center = int(x+w/2)
					text_r = self.TextRegions_[i][:,max(center-20,0):min(center+20,n)]
					# cv2.rectangle(draw,box,color=(255))
					text[i].append(text_r)

		return text
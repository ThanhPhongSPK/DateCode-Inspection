import cv2
import numpy as np
from scipy.signal import argrelextrema
from math import *
def gaussian_filter(size):
    sigma = size/4
    # Ensure the size is odd for symmetry
    if size % 2 == 0:
        size += 1

    # Create the filter
    midpoint = size // 2
    indices = np.arange(size)
    filter_array = np.exp(-(indices - midpoint)**2 / (2 * sigma**2))
    filter_array /= np.sum(filter_array)  # Normalize to sum to 1

    return filter_array
def hsv_equalizedV2(BGRimage,CLAHE = False ):
    if not CLAHE:
        H, S, V = cv2.split(cv2.cvtColor(BGRimage, cv2.COLOR_BGR2HSV))
        eq_S = cv2.equalizeHist(S)
        eq_image = cv2.cvtColor(cv2.merge([H, eq_S, V]), cv2.COLOR_HSV2RGB)
    else:
        lab= cv2.cvtColor(BGRimage, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        eq_image =cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return eq_image
def GetTheBoundingBox(cnts,listBox,size):
	if len(listBox)>0:
		a  = np.zeros(a)
		for box in listBox:
			x,y,w,h = box
			gauss = gaussian_filter(w)*w*h
			a[x:x+w] = a[x:x+w]+gauss
		a = a/np.mean(a)
		idx = np.where(a < 0.1)
		c =  np.diff(idx)
		list_edge = np.where(c>1)[0]
		border = [idx[i] for i in list_edge]
		border = np.array(border).reshape(-1,1)
		border = np.hstack([border[1:],border[:-1]])
		listCnt = [[] for b in border]
		for i,box in enumerate(listBox):
			x,y,w,h = box
			check = border-x-w/2
			check = check[:,0]*check[:,1]
			idx = np.where(check<0)[0]
			if idx:
				listCnt

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
		self.list_cen = [np.array([20,48,69,93,129,158,186,230,261,285,312,364,390,416,438,460,500,522,550]),
				   		np.array([15,36,64,86,123,153,183,228,262,306,339,379,404,428,452])]
		# self.up_number_ofWord = 
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
		try:
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
			#get the bounding rectangle of all the big distribution contours
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
			
			# TAKE SOME CHECKER
			up_checker = np.where(up_checker>150)[0].tolist()
			down_checker = np.where(down_checker>150)[0].tolist()
		
			up_checker1 = up_checker[0]
			down_checker1 = down_checker[0]
			up_checker2 = up_checker[-1]
			down_checker2 = down_checker[-1]
			mid_point_finder1 = int((up_checker1 + down_checker1)/2)
			
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
			angle_r = np.arctan2(p[1],p[0])*180/pi
			gray_r = RotateDegree(img=gray_r, angle_degree=angle_r, center=(width/2,height/2), size=(width,height))

			#PREPROCESSING_ CONVERT FOR EASY USING
			# gamma = 0.4
			# gray_c = (np.power((gray_r)/255,gamma)*255).astype(np.uint8)
			# thresh = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
			# 								cv2.THRESH_BINARY_INV,15, 5)
			gray_r = cv2.resize(gray_r,(560,100),interpolation = cv2.INTER_LINEAR)
			# thresh = cv2.resize(thresh,(560,100),interpolation = cv2.INTER_LINEAR)
			thresh = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
									cv2.THRESH_BINARY_INV,15, 5)
			kernel = np.ones((2, 2), np.uint8)
			thresh = cv2.erode(thresh,kernel,iterations = 1)
			
			biengioi = np.sum(thresh[45:55,:],axis=1)
			biengioi = np.where(biengioi<=5100)[0] + 45
			# print(biengioi[0])
			up 	= 	thresh[biengioi[0]-40:biengioi[0],:]
			down = 	thresh[biengioi[-1]+3:min(biengioi[-1]+43,100),:]
			up_color = gray_r[biengioi[0]-40:biengioi[0],:]
			down_color = gray_r[biengioi[-1]+3:min(biengioi[-1]+43,100),:]
			# FLIP IF NECCESSARY
			if up_checker1 > down_checker1:
				up = cv2.flip(up,1)
				up = cv2.flip(up,0)
				down = cv2.flip(down,1)
				down = cv2.flip(down,0)
				up_color = cv2.flip(up_color,1)
				up_color = cv2.flip(up_color,0)
				down_color = cv2.flip(down_color,1)
				down_color = cv2.flip(down_color,0)
				down_checker1 = up_checker1
			down = 	down[:,2*down_checker1-4:-2*down_checker1]
			down_color = down_color[:,2*down_checker1-4:-2*down_checker1]

			up = np.pad(up,((0,0),(0,2)),'constant',constant_values=0)
			down = np.pad(down,((0,0),(0,2)),'constant',constant_values=0)
			
			self.TextRegions_ = [up_color,down_color]
			return gray_r,up,down
		except Exception as e:
			print(e)
			return None,None,None
		
	def GetDistribution(self,text_region, lower_line = True):
		# text_region = cv2.bitwise_not(text_region)
		distribute = np.mean(text_region,axis=0)
		filter = np.ones(6)/6
		distribute_conv = np.convolve(filter,distribute,mode='same')
		f = distribute - distribute_conv
		filter = np.array((0.25,0.5,0.25))
		f = np.convolve(filter,f,mode='same')
		n = len(f)
		fhat = np.fft.fft(f,n)                     # Compute the FFT
		## Use the PSD to filter out noise
		if not lower_line:
			indices = (np.arange(n)<25)*(np.arange(n)>2)
		else:
			indices = (np.arange(n)<20)*(np.arange(n)>2)
		fhat = indices * fhat     # Zero out small Fourier coeffs. in Y
		ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered time signal
		return ffilt,distribute_conv
	
	def TextExtracting(self,text_regions):
		text = [[],[]]
		tR =[[],[]]
		for i in range(2):
			text_region = text_regions[i]
			m,n = text_region.shape
			kernel = np.ones((3,3))
			thresh_eroded = cv2.dilate(text_region , kernel, iterations=1)
			down_size = cv2.resize(thresh_eroded,(240,12),cv2.INTER_CUBIC)

			ffilt, distribute_conv = self.GetDistribution(down_size, lower_line = i)
			# for local minima
			list_pos = argrelextrema(ffilt, np.less)[0] # contain the region of text throughout x axis    
			list_pos = (list_pos*n/240).astype(int)
			# for zero region detectin only:
			distribute = np.mean(text_region,axis=0)
			a = np.where(distribute<15)[0]
			c = np.diff(a)
			kernel = np.array((0.2,0.6,0.2))
			cc = np.convolve(kernel,c,mode='same')
			change = np.where(cc<=3)[0]
			idx = [a[j] for j in change]
			x_mask1 = np.zeros(n)
			x_mask2 = np.zeros(n)
			for boundary in list_pos:
				x_mask1[boundary-5:boundary+8]=1
			for boundary in idx:
				x_mask2[boundary-2:boundary+3]=1
		
			mask = cv2.bitwise_and(x_mask1,x_mask2)
	
			mask = np.where(mask==True)[0]

			kernel = np.ones((3,1))
			thresh_eroded = cv2.dilate(text_region , kernel, iterations=2)
			# thresh_eroded = text_region.copy()
			for boundary in mask:
				thresh_eroded[:,boundary] = 0

			list_edge = np.diff(mask)
			list_edge = np.where(list_edge>1)[0]
			border = [0]+[mask[idx] for idx in list_edge]+[n]
			border = np.array(border).reshape(-1,1)
			border = np.hstack([border[:-1,:],border[1:,:]])
		
			edged = cv2.Canny(thresh_eroded, 100, 200)
			contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

			draw = edged.copy()

			list_boxes = [[] for c in self.list_cen[i]]
			list_cen = [c for c in self.list_cen[i]]
			for p in border:
				x = int((p[0]+p[1])/2)
				w = p[0]-p[1]
				if w <30:
					checker1 = (self.list_cen[i] - x +3)>0
					checker2 = (x - self.list_cen[i]+3)>0
					index = np.where((checker1*checker2)==True)[0]
					if len(index) >= 1:
						list_cen[index[0]] = (list_cen[index[0]]+x)/2
			list_cen = np.array(list_cen) 

			for cnt in contours:
				box = cv2.boundingRect(cnt)
				# cv2.rectangle(draw,box,color=(255))
				x,y,w,h= box
				if w*h > 50:
					cv2.rectangle(draw,box,color=(255))	
					x,y,w,h = [x+2,y+2,w+4,h+4] 
					center = np.array([x+w/2,y+h/2])

					checker1 = (list_cen - x+6)>0
					checker2 = (x + w - list_cen+6)>0

					
					index = np.where((checker1*checker2)==True)[0]
					if len(index) >= 1:
						minimum = 100
						k = index[0]
						for k_ in index:
							dist = np.abs(list_cen[k_]-x)
							if dist < minimum:
								minimum = dist
								k = k_

						list_boxes[k].append(cnt)
				
			
			for cnts in list_boxes:

				if len(cnts)==0:
					continue
				cnt = np.vstack(cnts)
				box = cv2.boundingRect(cnt)
				
			# 	box = GetTheBoundingBox(list_box)
				crop = None
				if box:
					xo,yo,wo,ho = box
					
					center = int(xo+wo/2)	
					text_r = self.TextRegions_[i][:,max(center-12,0):min(center+12,n)]
					text[i].append(text_r)
	
			# tR.append(draw)
			# FOR VISUALIZE THE RESULT

			for i,lT in enumerate(text):	
				if len(lT)>0:
					listT = []
					for t in lT:
						if t is not None:
							# print('i',i,t.shape)
							t = np.pad(t,4,mode='constant',constant_values=0)
							listT.append(t)
					tR[i] = np.concatenate(listT,axis=1)
			
		return text[0],text[1] # return list of text
		# return tR[0],tR[1],draw
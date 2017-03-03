import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Threshold:

	def __init__(self):
		self.ksize = 15 # Sobel kernel size
		#self.sobel_grad_thresh = (20, 150)
		self.sobel_grad_thresh =  (50, 150)
		self.combined_binary = None
		self.color_binary = None
	
	def color_threshold(self, img, s_thresh=(170, 255)):
		# Convert to HSV color space and separate the V channel
		hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
		s_channel = hsv[:,:,2]
		
		# Threshold color channel
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
		return s_binary
    

	def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
		
		# Apply the following steps to img
		# 1) Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# 2) Take the derivative in x or y given orient = 'x' or 'y'
		
		if(orient == 'x'):
			sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0, ksize=sobel_kernel)
		else:
			sobelx = cv2.Sobel(gray,cv2.CV_64F,0,1, ksize=sobel_kernel)
		# 3) Take the absolute value of the derivative or gradient
		abs = np.absolute(sobelx)
		# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
		scaled = np.uint8(255*abs/np.max(abs))
		# 5) Create a mask of 1's where the scaled gradient magnitude 
		# is > thresh_min and < thresh_max
		binary = np.zeros_like(scaled)
		# 6) Return this mask as your binary_output image
		binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
		return binary
		
	def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
		
		# Apply the following steps to img
		# 1) Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# 2) Take the gradient in x and y separately
		sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray,cv2.CV_64F,0,1, ksize=sobel_kernel)
		# 3) Calculate the magnitude
		abs = np.sqrt(sobelx**2 + sobely**2)
		# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
		scaled = np.uint8(255*abs/np.max(abs))
		# 5) Create a binary mask where mag thresholds are met
		binary = np.zeros_like(scaled)
		binary[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
		# 6) Return this mask as your binary_output image
		return binary
		
	def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
		
		# Apply the following steps to img
		# 1) Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# 2) Take the gradient in x and y separately
		sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray,cv2.CV_64F,0,1, ksize=sobel_kernel)
		# 3) Take the absolute value of the x and y gradients
		#abs = np.sqrt(sobelx ** 2 + sobely **2)
		abs_sobely = np.absolute(sobely)
		abs_sobelx = np.absolute(sobelx)
		# 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
		arctan = np.arctan2(abs_sobely, abs_sobelx)
		# 5) Create a binary mask where direction thresholds are met
		binary = np.zeros_like(arctan)
		binary[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 1
		# 6) Return this mask as your binary_output image
		return binary
		
	def get_threshold(self, img):
		
		# Apply each of the thresholding functions

		gradx = self.abs_sobel_thresh(img, orient='x', sobel_kernel=self.ksize, thresh=self.sobel_grad_thresh)
		grady = self.abs_sobel_thresh(img, orient='y', sobel_kernel=self.ksize, thresh=self.sobel_grad_thresh)
		#mag_binary = self.mag_thresh(img, sobel_kernel=self.ksize, mag_thresh=(30, 100))
		#dir_binary = self.dir_threshold(img, sobel_kernel=self.ksize, thresh=(0.7, 1.3))
		#s_binary = self.color_threshold(img, s_thresh=(90, 255))
		
		mag_binary = self.mag_thresh(img, sobel_kernel=self.ksize, mag_thresh=(30, 100))
		dir_binary = self.dir_threshold(img, sobel_kernel=self.ksize, thresh=(0, np.pi/2))
		s_binary = self.color_threshold(img, s_thresh=(90, 150))


		#combined_binary = np.zeros_like(dir_binary)
		#combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

		#Combine the Thresholds
		self.combined_binary = np.zeros_like(dir_binary)
		#self.combined_binary[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | ((s_binary == 1))] = 1
		self.combined_binary[(gradx == 1) | (grady == 1) | (mag_binary == 1) & (dir_binary == 1) | (s_binary == 1)] = 1
		self.color_binary = np.dstack(( np.zeros_like(self.combined_binary), self.combined_binary, s_binary))
		
		return self.combined_binary
		
	def display(self, img, show=False):
		if show == True:
			#img = mpimg.imread('test_images/test5.jpg')
			self.get_threshold(img)

			f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
			f.tight_layout()
			ax1.imshow(img)
			ax1.set_title('Original Image', fontsize=50)
			ax2.imshow(self.color_binary, cmap='gray')
			ax2.set_title('Stacked thresholds.', fontsize=50)
			ax3.imshow(self.combined_binary, cmap='gray')
			ax3.set_title('Combined thresholds.', fontsize=50)	
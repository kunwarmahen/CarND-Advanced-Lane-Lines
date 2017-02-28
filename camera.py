import glob 
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Camera:

	def __init__(self):
		self.mtx = None;
		self.dist = None;
	
	def calibirateCamera(self):
		
		# Read in all the calibration images
		images = glob.glob('camera_cal/calibration*.jpg')

		objpoints = []
		imgpoints = []
		nx = 9
		ny = 6
		objp = np.zeros((ny*nx, 3), np.float32)
		objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

		for fname in images:
			img = mpimg.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

			if ret==True:  
				imgpoints.append(corners)
				objpoints.append(objp)
				cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
		
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	def undistort(self,img):
		return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
		
	def display(self, show=False):
		if show == True:
			img = cv2.imread('camera_cal/calibration2.jpg')
			undst = self.undistort(img)
			f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
			ax1.imshow(img)
			ax1.set_title('Original Image', fontsize=30)
			ax2.imshow(undst)
			ax2.set_title('Undistorted Image', fontsize=30)	
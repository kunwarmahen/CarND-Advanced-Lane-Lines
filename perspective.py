import cv2
import numpy as np
import matplotlib.pyplot as plt

class Perspective:

	def __init__(self):
		#self.src = np.float32([[571,460],[700, 460],[1034,673],[276,673]]) #large
		self.dst = np.float32([[264,0],[1034, 0],[1034,679],[264,679]])	
		#self.src = np.float32([[534,495],[748, 490],[1035,675],[256,685]]) # small
		#self.dst = np.float32([[264,126],[1034, 126],[1034,679],[264,679]])	
		self.src = np.float32([[559,476],[727, 475],[1037,673],[255,684]]) #middle

	def perspective_transform(self):
		return cv2.getPerspectiveTransform(self.src, self.dst) 

	def inverse_perspective_transform(self):
		return cv2.getPerspectiveTransform(self.dst, self.src) 

	def warp_image(self, img, M):
		return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
		
	def display(self, img, warped, show=False):
		if show == True:
			f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
			f.tight_layout()
			ax1.imshow(self.warp_image(img, self.perspective_transform()))
			ax1.set_title('Original Image Warped', fontsize=50)
			ax2.imshow(warped, cmap='gray')
			ax2.set_title('Binary Image Warped.', fontsize=50)
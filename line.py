import cv2
import numpy as np
from perspective import Perspective
import matplotlib.pyplot as plt

class Line:
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False  
		# x values of the last n fits of the line
		self.recent_xfitted = [] 
		#average x values of the fitted line over the last n iterations
		self.bestx = None     
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = None  
		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]  
		#radius of curvature of the line in some units
		self.radius_of_curvature = None 
		#distance in meters of vehicle center from the line
		self.line_base_pos = None 
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float') 
		#x values for detected line pixels
		self.allx = None  
		#y values for detected line pixels
		self.ally = None
		
		self.left_fitx = None
		self.right_fitx = None
		self.leftx = None
		self.lefty = None
		self.rightx = None
		self.righty = None
		self.position = None
		self.left_curverad = None
		self.right_curverad = None
		
		
		
		# Define conversions in x and y from pixels space to meters
		self.ym_per_pix = 30/720 # meters per pixel in y dimension
		self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
		
	def detect_initial_lane_lines(self, warped, display=False):
		histogram = np.sum(warped[warped.shape[0]/2:,:], axis=0)
		
		if display == True:
			plt.figure()
			plt.plot(histogram)
			out_img = np.dstack((warped, warped, warped))*255

		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 9
		# Set height of windows
		window_height = np.int(warped.shape[0]/nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = 100
		# Set minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = warped.shape[0] - (window+1)*window_height
			win_y_high = warped.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			if display == True:
				cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
				cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		self.leftx = nonzerox[left_lane_inds]
		self.lefty = nonzeroy[left_lane_inds] 
		self.rightx = nonzerox[right_lane_inds]
		self.righty = nonzeroy[right_lane_inds] 

		# Fit a second order polynomial to each
		left_fit = np.polyfit(self.lefty, self.leftx, 2)
		right_fit = np.polyfit(self.righty, self.rightx, 2)

		# Generate x and y values for plotting
		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
		self.left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		self.right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		
		# Calculate the position of the vehicle
		self.position = ((np.mean(self.left_fitx)+np.mean(self.right_fitx))/2)
		
		if display == True:
			out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
			out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
			plt.figure()
			plt.imshow(out_img)
			plt.plot(self.left_fitx, ploty, color='yellow')
			plt.plot(self.right_fitx, ploty, color='yellow')
			plt.xlim(0, 1280)
			plt.ylim(720, 0)
    
	def detect_subsequent_lane_lines(self, warped):
		nonzero = warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100
		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

		# Again, extract left and right line pixel positions
		self.leftx = nonzerox[left_lane_inds]
		self.lefty = nonzeroy[left_lane_inds] 
		self.rightx = nonzerox[right_lane_inds]
		self.righty = nonzeroy[right_lane_inds]
		# Fit a second order polynomial to each
		left_fit = np.polyfit(self.lefty, self.leftx, 2)
		right_fit = np.polyfit(self.righty, self.rightx, 2)
		# Generate x and y values for plotting
		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
		self.left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		self.right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		
	def measure_curvature(self, warped, display=False):

		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
		y_eval = np.max(ploty)

		if display == True:
			# Plot up the data
			plt.figure()
			mark_size = 3
			plt.plot(self.leftx, self.lefty, 'o', color='red', markersize=mark_size)
			plt.plot(self.rightx, self.righty, 'o', color='blue', markersize=mark_size)
			plt.xlim(0, 1280)
			plt.ylim(0, 720)
			plt.plot(self.left_fitx, ploty, color='green', linewidth=3)
			plt.plot(self.right_fitx, ploty, color='green', linewidth=3)
			plt.gca().invert_yaxis() # to visualize as we do the images
	
		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(self.lefty*self.ym_per_pix, self.leftx*self.xm_per_pix, 2)
		right_fit_cr = np.polyfit(self.righty*self.ym_per_pix, self.rightx*self.xm_per_pix, 2)
		# Calculate the radii of curvature
		self.left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		self.right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		# Now our radius of curvature in meters
		
	def identify_lane_area(self, warped):
		# Create an image to draw the lines on
		warp_zero = np.zeros_like(warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([self.left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		perspective = Perspective()
		newwarp = perspective.warp_image(color_warp, perspective.inverse_perspective_transform())
		return newwarp
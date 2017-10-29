import numpy as np
import cv2
def watershed(imgname)	:


	img = imgname

	#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray=img
	img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=3)
	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1

	markers = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	markers = cv2.watershed(img,markers)
	img[markers == -1] = [255,0,0]
	cv2.imshow("final",img)
	cv2.imshow("sure_fg",sure_fg)
	cv2.imshow("sure_bg",sure_bg)
	k = cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()
	return;
def sharpen(img):
	im=img
	kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	im = cv2.filter2D(im, -1, kernel)
	return im;
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
	v = np.median(image)

	    # apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	    # return the edged image
	return edged
def blobdetect(img):
	im = img
	inp=img
	#im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		 
		# Set up the detector with default parameters.
	#imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(im,100,255,0)
	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
	
	#cv2.imshow("blob",blobdetect)
	mask = np.ones(im.shape[:2], dtype="uint8") * 255
 
	# loop over the contours
	for c in contours:
		# if the contour is bad draw it on the mask
		if cv2.contourArea(c)<150000:
			cv2.drawContours(mask, [c], -1, 0, -1)
			print(cv2.contourArea(c))
	 
	# remove the contours from the image and show the resulting images
	img2= cv2.bitwise_and(img, img, mask=mask)
	
	#cv2.imshow("input image",inp)
	#cv2.imshow("Mask", mask)
	#cv2.imshow("After", img2)
	cv2.imshow("contours formed",img)



	#.imshow('log',laplacian)
	k = cv2.waitKey(0) & 0xFF
	cv2.destroyAllWindows()


	return img2;


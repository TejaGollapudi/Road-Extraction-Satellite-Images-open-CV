import numpy as np
import cv2
from sklearn.metrics import accuracy_score
img = cv2.imread('ds3label.jpg')
img2=cv2.imread('ds3output.jpeg')
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
mask=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
(thresh, mask2) = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
output = cv2.bitwise_and(img, img, mask = mask)
print(np.maximum(mask.flatten(),img2.flatten()))
d=accuracy_score(mask.flatten(),img2.flatten())
print(d)
 
	# show the images
cv2.imshow("images", np.hstack([img, output]))
cv2.imshow("mask",mask2)
cv2.imshow("final",img2)
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
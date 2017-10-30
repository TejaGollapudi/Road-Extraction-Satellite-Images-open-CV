from water import watershed,sharpen,auto_canny,blobdetect
import numpy as np
import cv2


frame2 = cv2.imread('dataset/img1.jpg') #img load 
sharped=sharpen(frame2)

frame=cv2.medianBlur(frame2,5)
#black=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#lack2=cv2.GaussianBlur(black,(3,3),0)

#laplacian = cv2.Laplacian(black2,cv2.CV_64F)


'''

green = 60;
blue = 120;
yellow = 30;

sensitivity = 15
#lower = np.array([0, 0, 230])
#upper = np.array([180, 20, 255])
#lower = np.array([green- sensitivity, 100, 50]) 
#upper = np.array([green + sensitivity, 255, 255])
 

    # Threshold the HSV image to get only blue colors
#lower = np.array([70, 95, 82])
#upper = np.array([126, 100, 100])
lowerlight=np.array([48 ,15 ,140])
upperlight=np.array([65 ,25 ,225])
lowerdark=np.array([40 ,18 ,100])
upperdark=np.array([70 ,24 ,180])
lowersand=np.array([18 ,22 ,100])
uppersand=np.array([23 ,30 ,205])
lower=np.array([15,15,90])
upper=np.array([70,35,230])'''
'''maskother=cv2.inRange(hsv,lower,upper)

masklight= cv2.inRange(hsv,lowerlight,upperlight)
maskdark= cv2.inRange(hsv,lowerdark,upperdark)
masksand= cv2.inRange(hsv,lowersand,uppersand)
mask = cv2.bitwise_or(maskdark, masklight)
mask2=cv2.bitwise_or(mask,masksand)'''
'''
img=cv2.GaussianBlur(mask2,(5,5),0)
    # Bitwise-AND mask and original image
res = cv2.bitwise_and(frame2,frame2, mask= mask2)
res2=cv2.bitwise_and(frame2,frame2,mask=maskother)
kernel = np.ones((5,5), np.uint8)
'''

#################################################################################################






#find all your connected components (white blobs in your image)
#es2=res2.astype(np.uint8)
#output = cv2.connectedComponentsWithStats(res2, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
'''sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 150  

#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255


'''

####################################MORPHOLOGY CLOSING#################################################################
'''kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(res2, cv2.MORPH_CLOSE, kernel)'''
##########################################K-MEANS##############################################
Z = frame.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 20

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
print(ret)
resd = center[label.flatten()]
resd2 = resd.reshape((frame.shape))
cluster=resd2
resd2=cv2.cvtColor(resd2,cv2.COLOR_BGR2GRAY)
#retval, resd2 = cv2.threshold(resd2, 150, 255, cv2.THRESH_BINARY)


#resd2=cv2.medianBlur(resd2,5)
sharpcluster=sharpen(resd2)

##########################################canny edge####################################################
edges = cv2.Canny(frame,400,600)
#edges2=cv2.Canny(img_dilation,249,250)
edges3=cv2.Canny(resd,200,260)
edgesharped=auto_canny(sharpcluster)
###########################################smoothen kmeans output##################################################
smoothmeans=cv2.GaussianBlur(resd2,(5,5),0)
##########################################################histogram equalisation##########################################
equ = cv2.equalizeHist(resd2)


####################################BLOBDETECT######################
final=blobdetect(equ)

#########################################IMAGE errosion dilation 

'''img_erosion = cv2.erode(equ, kernel, iterations=1)

img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
'''


   ##################################IMAGE DISPLAY########################################################################################

#cv2.imshow('original',frame2)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)
#cv2.imshow('converted image',hsv)
#cv2.imshow('image',frame2)
#cv2.imshow('image',frame)
#cv2.imshow('masklight',masklight)
#cv2.imshow('maskdark',maskdark)
#cv2.imshow('masksand',masksand)
#cv2.imshow('final',mask2)
#cv2.imshow('blur',img)
#cv2.imshow('result',res)
#cv2.imshow('result2',res2)
#v2.imshow('edges',edges)
#cv2.imshow('edges2',edges2)
#cv2.imshow("removed small objects",img2)
#v2.imshow("closing",closing)
#cv2.imshow("hsv method 2",res2)
#cv2.imshow("hsv method 1",res)
#cv2.imshow("errorsion",img_erosion)
#cv2.imshow("dilation",img_dilation)
cv2.imshow("input image",frame2)
cv2.imshow("GaussianBlur output",frame)
cv2.imshow("clustering",cluster)
cv2.imshow("histogram",equ)
#watershed(res2)
#v2.imshow("sharped",sharpcluster)
cv2.imshow("contour output",final)
#cv2.imshow("auto canny of sharped edges of cluster",edgesharped)
#cv2.imshow("blob",blobdetect)
#watershed(final)
cv2.imshow("grey conversion",resd2)
retval, threshold = cv2.threshold(final, 220, 255, cv2.THRESH_BINARY)
cv2.imshow("final",threshold)
#watershed(threshold)

#.imshow('log',laplacian)
k = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
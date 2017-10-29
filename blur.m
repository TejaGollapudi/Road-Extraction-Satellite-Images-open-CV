I = imread('.\DevelopedUrban\es7.jpg');
figure
imshow(I)
hsv_map = rgb2hsv (I)

title('Original image')
Iblur1 = imsmooth(I, "Gaussian");


imshow(Iblur1)
title('Smoothed image')
thresholds = [5.2742e-004 0.3899];
gray = rgb2gray (Iblur1)
bh=fspecial("log")
figure

figure
imshow(gray)
BW = edge(gray, 'Canny');
figure
imshow(BW)


%---------------------------------------------------------------------------------




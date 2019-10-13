
# coding: utf-8

# In[139]:

# Cài imutils "pip install imutils"
import numpy as np
import imutils
import cv2

# Import ảnh và chuyển thành dạng xám
my_image = cv2.imread("05.jpg")
gray = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
 
# Giữ các vùng có high horizontal gradient và low vertical gradient
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gx = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gy = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
gd = cv2.subtract(gx, gy)
gd = cv2.convertScaleAbs(gd)

blurred = cv2.blur(gd, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)


# Chọn tham số và thresholding
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
image_temp = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Loại bỏ noise
image_temp = cv2.erode(image_temp, None, iterations = 4)
image_temp = cv2.dilate(image_temp, None, iterations = 4)

# cv2.imshow("Image", image_temp)
# cv2.waitKey(0)

# Tìm đường viền trong ảnh threshold, giữ lại vùng lớn nhất
cnts = cv2.findContours(image_temp.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
  
# Vẽ box xung quanh vùng chọn 
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)

cv2.drawContours(my_image, [box], -1, (0, 255, 0), 3)
cv2.imshow("5.jpg", my_image)
cv2.waitKey(0)


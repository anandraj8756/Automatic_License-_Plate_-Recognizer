

######################################
# Automatic License Plate Recognizer #
# Author: Anand raj                  #
# Email: anandallen95gmail.com       #
######################################


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract

img2 = cv2.imread('cropimage.jpg')
#This is only for numberplate image , If you are giving input only numberplate image
#then it will give output of detected number and text form of number

cv2.imshow('Original',img2)
#Noise Removel
nr_img = cv2.bilateralFilter(img2,11,17,17)

#Convert in gray image
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#cv2.imshow('GrayImg',gray)

soblex = cv2.Sobel(gray,cv2.CV_8U,1,0,ksize=3)
cv2.imshow('Solbex',soblex)

ret, thresh = cv2.threshold(gray,130,255, cv2.THRESH_TOZERO)
#cv2.imshow('Thresh',thresh)

ret, threshs = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#cv2.imshow('threshs',threshs)

element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5,11) )

morph_img= thresh.copy()
cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)
#cv2.imshow('morphimg',morph_img)


contours, hierarchy= cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for c in contours:
    r = cv2.boundingRect(c)
    cv2.rectangle(morph_img,(r[0],r[1]),(r[0]+r[2],r[1]+r[3]),(0,255,0))
    cv2.rectangle(nr_img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0))

    #cv2.imshow('Segimg',morph_img)
    cv2.imshow('seg2',nr_img)


text =pytesseract.image_to_string(nr_img, config='')
print(text)


cv2.waitKey(0)
cv2.destroyAllWindows()
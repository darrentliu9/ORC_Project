import cv2
import pytesseract
import numpy as np
import re
from pytesseract import Output
import time

startTime = time.time()
def read_text_from_image(image):
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # thresholding
  ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
  # constructing Convolutional Kernels
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
  # dilation
  dilation = cv2.dilate(thresh, kernel, iterations = 1)

  # Finding image contours in binary images
  contours, hierachy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  # Draw a border around the text
  cv2.drawContours(image, contours, -1, (0,255,0), 2)

  # Image preprocessing process (for program demonstration only)
  # cv2.imshow('gray_image',gray_image)
  # cv2.waitKey(0)
  # cv2.imshow('thresh',thresh)
  # cv2.waitKey(0)
  # cv2.imshow('dilation',dilation)
  # cv2.waitKey(0)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  # Extract text and put it into a txt file
  image_copy = image.copy()
  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image_copy[y : y + h, x : x + w]
    file = open("result.txt", "a")
    text = pytesseract.image_to_string(cropped)
    file.write(text)
    # file.write("\n")
  file.close()
  print(time.time() - startTime)

image = cv2.imread('image1.png')
# Adding custom options
custom_config = r'--oem 3 --psm 6'

pytesseract.image_to_string(image, config=custom_config)
read_text_from_image(image)


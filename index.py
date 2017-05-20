import os
import csv
import sys
import glob
import argparse
import json
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import itemfreq
from skimage.feature import hog , local_binary_pattern


#Apply a filename standard
def renameDataset(dataset):
	for (i, filename) in enumerate(dataset):
		current_image = cv2.imread(filename)
		cv2.imwrite('dataset/example_'+ str(i) +'.jpg', current_image)


#Load Regex Paths to be processed
def load_dataset():
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"--dataset",
		required = True,
		help = "Path to Images Directory, ex: abc/images/*.png"
	)

	args = vars(ap.parse_args())
	path = args["dataset"]

	dataset = [ file for file in glob.glob(path) ]
	return dataset


#Important
#Test few filters
#Sobel
#Canny
#Watershed

# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
def find_contours(image):
	#erosion
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
	grad = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
	#binarize
	_, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#dilation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
	dilated = cv2.dilate(thresh,kernel, iterations = 4) # dilate
	# connected = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
	cv2.imshow('dilated', dilated)
	_, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
	return contours




def drawRectangle(original, contours):
	for contour in contours:
		# get rectangle bounding contour
		[x, y, w, h] = cv2.boundingRect(contour)

		valid = original.shape[1] * 0.50
		#be smart and draw only in common signature regions
		# bootom of the figure
		# horizontal text
		if (w > h and y > valid and w > 20 and h > 20 ):
			# draw rectangle around contour on original image
			cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 255), 2)
			cv2.imshow('contours', original)

#Crop the image to commonly signature areas ( bottom left or bottom right)
def applyROI(image):
# 	width = image.shape[0]
# 	height = image.shape[1]
# 	width_ROI = math.floor(width * 0.30)
# 	height_ROI = math.floor(height * 0.30)

# 	#bottom left
# 	image = image[  width - width_ROI : width,  0 : height_ROI ]
	image = cv2.resize(image, (256, 256))
	return image



def process_images(dataset):
	print("Start Processing")
	for (i, imagePath) in enumerate(dataset):

		# Logging to follow uptime ( +- 5min to process)
		if i > 0 and i % 1000 == 0:
			print("Processed {} of {}".format(i, len(dataset)))
	print("End Processing")


def main(argv):

	# DATASET = load_dataset()
	# process_images(DATASET[0:2])
	image = cv2.imread('dataset/example_5.jpg', 0)
	image = applyROI(image)
	contours = find_contours(image)
	drawRectangle(image, contours)
	cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv)

# Get the pictures
# Apply some filters
# Find the signature contours in the picture
# Draw it and save the result image
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

RESULTS = []

#Apply a filename standard
def renameDataset(dataset):
	for (i, filename) in enumerate(dataset):
		current_image = cv2.imread(filename)
		cv2.imwrite('dataset/example_'+ str(i) +'.jpg', current_image)

# write results
def save_results(path):
	for i, image in enumerate(RESULTS):
		cv2.imwrite(path + 'results_'+ str(i) +'.jpg', image)

#Load Regex Paths to be processed
def load_dataset():
	ap = argparse.ArgumentParser()
	ap.add_argument(
		"--dataset",
		required = True,
		help = "Path to Images Directory, ex: abc/images/*.fileformat"
	)

	ap.add_argument(
		"--save",
		required = True,
		help = "Path to Save Results, ex: abc/images/ ! Important ! The path must exist"
	)

	args = vars(ap.parse_args())
	path = args["dataset"]
	writePath = args["save"]

	dataset = [ file for file in glob.glob(path) ]

	return dataset,writePath


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
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
	# dilated = cv2.dilate(thresh,kernel, iterations = 1) # dilate
	connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	_, contours, hierarchy = cv2.findContours(connected,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
	return contours


def drawRectangle(original, contours):
	for contour in contours:
		# get rectangle bounding contour
		[x, y, w, h] = cv2.boundingRect(contour)

		valid = original.shape[1] * 0.50
		#be smart and draw only in common signature regions
		# bootom of the figure
		# horizontal text
		# if (w > h and y > valid and w > 20 and h > 20 ):
			# draw rectangle around contour on original image
		marked = cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 255), 2)
		return marked

#Resize the image to do analysis easier
def applyROI(image):
	image = cv2.resize(image, (256, 256))
	return image

def process_images(dataset):
	print("Start Processing")
	for (i, imagePath) in enumerate(dataset):

		image = cv2.imread(imagePath, 0)
		image = applyROI(image)
		contours = find_contours(image)

		# save the processed image
		img_processed = drawRectangle(image, contours)
		RESULTS.append(img_processed)

		# Logging to follow uptime
		if i > 0 and i % 10 == 0:
			print("Processed {} of {}".format(i, len(dataset)))
	print("End Processing")


def main(argv):
	DATASET,WRITE_PATH = load_dataset()
	process_images(DATASET)
	save_results(WRITE_PATH)

if __name__ == "__main__":
    main(sys.argv)
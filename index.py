import os
import csv
import sys
import glob
import argparse
import json
import math

import cv2

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

# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# In this method we apply many morphological operations and
# filters to reduce noising of the image
# because our dataset its very sparse and mixed
def find_contours(image):
	# For the utilization of Otsu threshold its needed to apply a gaussian blur
	# we use a big kernel to instensify the noising remove
	blur = cv2.GaussianBlur(image,(11,11),0)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
	grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

	# Applying Otsu binarization its very useful because
	# he uses the advantage of a gaussian blured image
	# and calculates an optimal threshold based on the variance and means
	# wich is useful since we have many different images
	# http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
	_, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#dilation
	# we dilate to remove some noise resulted of the transformation
	dilated = cv2.dilate(thresh,kernel, iterations = 2) # dilate

	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
	connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	_, contours, hierarchy = cv2.findContours(connected,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
	cv2.imshow('contours', connected)
	cv2.waitKey(0)
	return contours


# apply the rectangle draw on the interesting regions
def drawRectangle(original, contours):
	for contour in contours:
		# get rectangle bounding contour
		[x, y, w, h] = cv2.boundingRect(contour)

		#being smart and draw only in commonly signature regions
		valid = original.shape[1] * 0.50
		# bootom of the figure
		# horizontal text
		if (w > h and y > valid and w > 20 and h > 20 ):
		# draw rectangle around the contour on the original image
			marked = cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 255), 2)
			return marked

def process_images(dataset):
	print("Start Processing")
	for (i, imagePath) in enumerate(dataset):
		# get the current gray image
		image = cv2.imread(imagePath, 0)
		# Find all contours in the gray image
		contours = find_contours(image)
		# Get the image with rectangles and save
		img_processed = drawRectangle(image, contours)
		cv2.imshow('contours', img_processed)
		cv2.imwrite('results/' + 'results_'+ str(i) +'.jpg', img_processed)

		# Logging to follow uptime
		if i > 0 and i % 10 == 0:
			print("Processed {} of {}".format(i, len(dataset)))
	print("End Processing")


def main(argv):
	DATASET,WRITE_PATH = load_dataset()
	process_images(DATASET)

if __name__ == "__main__":
    main(sys.argv)
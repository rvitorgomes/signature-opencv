import os
import csv
import sys
import glob
import argparse
import json

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

# Apply Gaussian Filter to reduce noise
# Apply Threshold to pixels inversion and inscrese difference between interested pixels from background
def apply_filters(image):
	blur = cv2.GaussianBlur(image,(5,5),0)
	ret, im_th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)
	return im_th

#Feature Extractions

#Local Binary Pattern
def feature_extractor_LBP(image):

	#Commom 3x3 kernel usage and 8 neighbohors
	radius = 3
	points = 8 * radius
	eps = 1e-7

	#Uniform LBP to increase rotation invariance
	lbp = local_binary_pattern(image, points, radius, method="uniform")

	(hist, bins) = np.histogram(lbp.ravel(),
		bins=np.arange(0, 256),
		range=(0, 256))

	#Calculate the histogram
	hist = hist.astype("float")
	#Normalization
	hist /= (hist.sum() + eps)

	return hist

#HOG with optimized pixels_per_cell that extract most relevant information
def feature_extractor_HOG(image):
	features = hog(
		image,
		orientations=9,
		pixels_per_cell=(16,16),
        cells_per_block=(1, 1),
		visualise=False
	)
	return features


def process_images(dataset):
	print("Start Processing")
	for (i, imagePath) in enumerate(dataset):

		# get the char class from dictionary
		label = getLabelClass(label)

		# applying the filters on the image
		image = cv2.imread(imagePath, 0)
		filtered_image = apply_filters(image)

		# calculate feature extraction for each raw
		hog = feature_extractor_HOG(filtered_image)
		lbp = feature_extractor_LBP(filtered_image)

		# update responses
		processed_data_hog[label].append(hog)
		processed_data_lbp[label].append(lbp)

		# Logging to follow uptime ( +- 5min to process)
		if i > 0 and i % 1000 == 0:
			print("Processed {} of {}".format(i, len(dataset)))
	print("End Processing")


def main(argv):

	DATASET = load_dataset()

	cv2.waitKey(0)

if __name__ == "__main__":
    main(sys.argv)

# Get the pictures
# Apply some filters
# Find the signature contours in the picture
# Draw it and save the result image
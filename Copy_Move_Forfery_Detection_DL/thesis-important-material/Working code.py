SIFT

# -*- coding: utf-8 -*-

import cv2 

#sift
sift = cv2.SIFT_create()

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


img1 = cv2.imread('1.jpg')  
img2 = cv2.imread('2.jpg') 

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)


img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:600], img2, flags=2)

cv2.imshow('SIFT', img3)


cv2.waitKey(0) 

=====================================

SURF

# -*- coding: utf-8 -*-


import cv2

# Load the image
image = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
cv2.imshow('Image with Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()



SURF AND SIFT

# -*- coding: utf-8 -*-


import cv2

# Load the image
image = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
cv2.imshow('Image with Keypoints (SIFT)', image_with_keypoints)

# Feature matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

img1 = gray  # The grayscale image with keypoints and descriptors already computed
img2 = cv2.imread('2.jpg')  # Load the second image
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)

matches = bf.match(descriptors, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)

matching_result = cv2.drawMatches(gray, keypoints, gray2, keypoints_2, matches[:100], None, flags=2)

# Display the feature matching result
cv2.imshow('SIFT Feature Matching', matching_result)

cv2.waitKey(0)
cv2.destroyAllWindows()




===========================


SURF SIFT and aglometric

# -*- coding: utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Load the image
image = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
cv2.imshow('Image with Keypoints (SIFT)', image_with_keypoints)

# Feature matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

img1 = gray  # The grayscale image with keypoints and descriptors already computed

# Generate some example data
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# Perform Agglomerative Hierarchical Clustering
linked = linkage(X, 'single')  # You can change the linkage type (e.g., 'single', 'complete', 'average', etc.)

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Load the second image
img2 = cv2.imread('2.jpg')

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)

matches = bf.match(descriptors, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)

matching_result = cv2.drawMatches(gray, keypoints, gray2, keypoints_2, matches[:100], None, flags=2)

# Display the feature matching result
cv2.imshow('SIFT Feature Matching', matching_result)

cv2.waitKey(0)
cv2.destroyAllWindows()


========================================================================

SIFT & SURF & Aglometric clustering & RANSAC


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Load the image
image = cv2.imread('1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
cv2.imshow('Image with Keypoints (SIFT)', image_with_keypoints)

# Feature matching using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

img1 = gray  # The grayscale image with keypoints and descriptors already computed

# Generate some example data
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# Perform Agglomerative Hierarchical Clustering
linked = linkage(X, 'single')  # You can change the linkage type (e.g., 'single', 'complete', 'average', etc.)

# Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Load the second image
img2 = cv2.imread('2.jpg')

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)

matches = bf.match(descriptors, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)

matching_result = cv2.drawMatches(gray, keypoints, gray2, keypoints_2, matches[:100], None, flags=2)

# Display the feature matching result
cv2.imshow('SIFT Feature Matching', matching_result)

# Convert keypoints to numpy arrays
points1 = np.array([keypoints[match.queryIdx].pt for match in matches])
points2 = np.array([keypoints_2[match.trainIdx].pt for match in matches])

# Perform RANSAC linear regression
def fit_line_ransac(x, y, iterations=100, threshold=3.0):
    best_model = None
    best_inliers = None
    best_num_inliers = 0

    for _ in range(iterations):
        sample_indices = np.random.choice(len(x), 2, replace=False)
        sample_x = x[sample_indices]
        sample_y = y[sample_indices]

        model_params = np.polyfit(sample_x, sample_y, 1)
        model = np.poly1d(model_params)

        distances = np.abs(y - model(x))
        inliers = np.where(distances < threshold)[0]
        num_inliers = len(inliers)

        if num_inliers > best_num_inliers:
            best_model = model
            best_inliers = inliers
            best_num_inliers = num_inliers

    return best_model, best_inliers

# Apply RANSAC to the matching keypoints
model, inliers = fit_line_ransac(points1[:, 0], points2[:, 0])

# Plot the original matching keypoints
plt.scatter(points1[:, 0], points2[:, 0], color='b', label='Matching Keypoints')

# Plot the RANSAC fitted line
plt.plot(points1[inliers, 0], model(points1[inliers, 0]), color='r', label='RANSAC Fitted Line')

plt.xlabel('X of Image 1')
plt.ylabel('X of Image 2')
plt.legend()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

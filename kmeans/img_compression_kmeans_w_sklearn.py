# import required library
import os, sys
import numpy as np
from sklearn.cluster import KMeans

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def apply_kmeans(image, n_clusters):
    # check if the image in a structure where we can process
    if len(image.shape) != 3:
        print("Image is not in the expected format. Expected shape is (height, width, 3)")
        return None

    # convert image to (height*width, 3)
    pixels = image.reshape(-1, 3)
    # apply kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=1453).fit(pixels)
    # get the labels
    labels = kmeans.labels_
    # get the centroids
    centroids = kmeans.cluster_centers_
    # convert the centroids to uint8
    centroids = centroids.astype(np.uint8)

    # Apply kmeans to the original image
    pixels_transformed = kmeans.transform(pixels)
    pixels_predicted = kmeans.predict(pixels)
    image_compressed = kmeans.cluster_centers_[pixels_predicted]
    image_quantized = image_compressed.reshape(image.shape).astype(np.uint8)
    return image_quantized, kmeans

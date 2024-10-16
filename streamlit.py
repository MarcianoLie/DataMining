import streamlit as st
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Function to resize image to a specific size
def resize_image(image, target_size=(256, 256)):
    return image.resize(target_size)

# Normalize pixel values to [0, 1]
def normalize_pixels(image_array):
    return image_array / 255.0

# Initialize random centroids for K-Means
def initialize_centroids(data, k):
    indices = random.sample(range(len(data)), k)
    centroids = data[indices]
    return centroids

# Assign data points to the nearest centroid
def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    return clusters

# Update centroids by calculating the mean of the assigned points
def update_centroids(data, clusters, centroids, k):
    new_centroids = []
    for cluster_idx in range(k):
        assigned_points = data[clusters == cluster_idx]
        if len(assigned_points) > 0:
            new_centroids.append(np.mean(assigned_points, axis=0))
        else:
            new_centroids.append(centroids[cluster_idx])
    return np.array(new_centroids)

# K-Means Clustering algorithm
def kmeans_clustering(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, centroids, k)
        
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

# Perform clustering on a new image using already trained centroids
def cluster_new_image_with_trained_model(img, centroids, resize_to=(256, 256)):
    img = resize_image(img, target_size=resize_to)  # Resize the image
    img = img.convert('RGB')  # Ensure it's RGB
    img_data = np.array(img)  # Convert image to NumPy array
    
    h, w, _ = img_data.shape  # Ensure the image has 3 channels (RGB)
    
    flattened_pixels = img_data.reshape(h * w, 3)  # Now (N, 3) shape
    flattened_pixels = normalize_pixels(flattened_pixels)  # Normalize the pixel values
    
    clusters = assign_clusters(flattened_pixels, centroids)
    
    clustered_image = np.zeros_like(img_data)
    for i in range(h * w):
        cluster_idx = clusters[i]
        clustered_image[i // w, i % w] = (centroids[cluster_idx] * 255).astype(np.uint8)  # Scale back to [0, 255]
    
    return img, clustered_image

# Streamlit app
def main():
    st.title("Image Clustering with K-Means")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Select number of clusters
        k = st.slider("Select number of clusters", min_value=2, max_value=5, value=4)

        # Process the image
        if st.button("Cluster Image"):
            st.write("Clustering in progress...")
            
            # Resizing and preparing image data
            img_data = np.array(resize_image(image, target_size=(256, 256)))
            img_data = normalize_pixels(img_data)
            h, w, _ = img_data.shape
            flattened_pixels = img_data.reshape(h * w, 3)
            
            # Train K-Means
            clusters, centroids = kmeans_clustering(flattened_pixels, k)
            
            # Apply clustering to the image
            _, clustered_image = cluster_new_image_with_trained_model(image, centroids, resize_to=(256, 256))
            
            # Display clustered image
            st.image(clustered_image, caption=f"Clustered Image with {k} clusters", use_column_width=True)

main()

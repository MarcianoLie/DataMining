import streamlit as st
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import time

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
def cluster_image_with_trained_model(img, centroids, resize_to=(256, 256)):
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
    
    return clustered_image

# Function to calculate silhouette score with 10% data sampling for faster calculation
def calculate_silhouette_score_fast(data, clusters, sample_ratio=0.1):
    sample_size = int(sample_ratio * len(data))  # Take 10% of the data for faster calculation
    sample_indices = random.sample(range(len(data)), sample_size)
    sampled_data = data[sample_indices]
    sampled_clusters = clusters[sample_indices]

    unique_clusters = np.unique(sampled_clusters)
    
    # Only compute silhouette score if more than 1 cluster exists
    if len(unique_clusters) > 1:
        silhouette_start_time = time.time()  # Track time for silhouette score calculation
        silhouette_avg = silhouette_score(sampled_data, sampled_clusters)
        silhouette_end_time = time.time()  # End time tracking
        
        print(f"Silhouette Coefficient for {len(unique_clusters)} clusters: {silhouette_avg}")
        print(f"Silhouette calculation took {silhouette_end_time - silhouette_start_time:.2f} seconds")
        return silhouette_avg
    else:
        print(f"Cannot calculate Silhouette Score, only {len(unique_clusters)} cluster(s) found.")
        return None

# Main function to train a K-Means model on the dataset and apply it to new images
def train_and_cluster_images(images, k):
    all_flattened_pixels = []

    # Resize images, extract features, and prepare data
    for image in images:
        img = resize_image(image, target_size=(256, 256))  # Resize the image
        img = img.convert('RGB')  # Ensure it's RGB
        img_data = np.array(img)
        img_data = normalize_pixels(img_data)  # Normalize pixel values to [0, 1]
        h, w, _ = img_data.shape  # Ensure the image has 3 channels (RGB)

        flattened_pixels = img_data.reshape(h * w, 3)  # Flatten image pixels into (N, 3)
        all_flattened_pixels.append(flattened_pixels)

    data = np.vstack(all_flattened_pixels)  # Combine all flattened pixel data for training
    
    # Train K-Means on the entire dataset
    clusters, centroids = kmeans_clustering(data, k)

    return clusters, centroids, data

# Streamlit app
def main():
    st.title("Image Clustering with K-Means")

    # Upload images
    uploaded_files = st.file_uploader("Choose images...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        # Select number of clusters
        k = st.slider("Select number of clusters", min_value=2, max_value=5, value=3)

        # Load images
        images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]

        # Display original images
        st.subheader("Original Images")
        for image in images:
            st.image(image, caption="Original Image", use_column_width=True)

        # Process all uploaded images together
        if st.button("Cluster Images"):
            st.write("Clustering in progress...")

            # Train K-Means and get clustered images
            clusters, centroids, data = train_and_cluster_images(images, k)

            # Display clustered images
            clustered_images = []
            for image in images:
                clustered_image = cluster_image_with_trained_model(image, centroids, resize_to=(256, 256))
                clustered_images.append(clustered_image)
                st.image(clustered_image, caption="Clustered Image", use_column_width=True)

            # Calculate silhouette score with faster method (10% sampling)
            with st.spinner("Calculating Silhouette Score..."):
                silhouette_avg = calculate_silhouette_score_fast(data, clusters)

            # Display silhouette score
            if silhouette_avg:
                st.write(f"Silhouette Score for the clustered images: {silhouette_avg:.2f}")
            else:
                st.write("Silhouette Score cannot be calculated due to insufficient clusters.")

if __name__ == "__main__":
    main()
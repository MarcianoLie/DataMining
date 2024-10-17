import streamlit as st
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import time

# Fungsi untuk mengubah ukuran gambar menjadi ukuran tertentu
def resize_image(image, target_size=(256, 256)):
    return image.resize(target_size)

# Normalisasi nilai piksel ke [0, 1]
def normalize_pixels(image_array):
    return image_array / 255.0

# Inisialisasi centroid acak untuk K-Means
def initialize_centroids(data, k):
    indices = random.sample(range(len(data)), k)
    centroids = data[indices]
    return centroids

# Menetapkan data poin ke centroid terdekat
def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
    clusters = np.argmin(distances, axis=1)
    return clusters

# Memperbarui centroid dengan menghitung rata-rata poin yang ditetapkan
def update_centroids(data, clusters, centroids, k):
    new_centroids = []
    for cluster_idx in range(k):
        assigned_points = data[clusters == cluster_idx]
        if len(assigned_points) > 0:
            new_centroids.append(np.mean(assigned_points, axis=0))
        else:
            new_centroids.append(centroids[cluster_idx])
    return np.array(new_centroids)

# Algoritma K-Means Clustering
def kmeans_clustering(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    
    for iteration in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, centroids, k)
        
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
    
    return clusters, centroids

# Melakukan clustering pada gambar baru menggunakan centroid yang sudah dilatih
def cluster_image_with_trained_model(img, centroids, resize_to=(256, 256)):
    img = resize_image(img, target_size=resize_to)  # Ubah ukuran gambar
    img = img.convert('RGB')  # Pastikan gambar dalam format RGB
    img_data = np.array(img)  # Ubah gambar menjadi array NumPy
    
    h, w, _ = img_data.shape  # Pastikan gambar memiliki 3 channel (RGB)
    
    flattened_pixels = img_data.reshape(h * w, 3)  # Sekarang dalam bentuk (N, 3)
    flattened_pixels = normalize_pixels(flattened_pixels)  # Normalisasi nilai piksel
    
    clusters = assign_clusters(flattened_pixels, centroids)
    
    clustered_image = np.zeros_like(img_data)
    for i in range(h * w):
        cluster_idx = clusters[i]
        clustered_image[i // w, i % w] = (centroids[cluster_idx] * 255).astype(np.uint8)  # Ubah kembali ke skala [0, 255]
    
    return clustered_image

# Fungsi untuk menghitung nilai silhouette dengan pengambilan sampel data 10% untuk perhitungan lebih cepat
def calculate_silhouette_score_fast(data, clusters, sample_ratio=0.1):
    sample_size = int(sample_ratio * len(data))  # Ambil 10% dari data untuk perhitungan lebih cepat
    sample_indices = random.sample(range(len(data)), sample_size)
    sampled_data = data[sample_indices]
    sampled_clusters = clusters[sample_indices]

    unique_clusters = np.unique(sampled_clusters)
    
    # Hanya hitung nilai silhouette jika terdapat lebih dari 1 cluster
    if len(unique_clusters) > 1:
        silhouette_start_time = time.time()  # Lacak waktu untuk perhitungan silhouette
        silhouette_avg = silhouette_score(sampled_data, sampled_clusters)
        silhouette_end_time = time.time()  # Akhiri pelacakan waktu
        
        print(f"Koefisien Silhouette untuk {len(unique_clusters)} cluster: {silhouette_avg}")
        print(f"Perhitungan silhouette memakan waktu {silhouette_end_time - silhouette_start_time:.2f} detik")
        return silhouette_avg
    else:
        print(f"Tidak dapat menghitung Skor Silhouette, hanya ditemukan {len(unique_clusters)} cluster.")
        return None

# Fungsi utama untuk melatih model K-Means pada dataset dan menerapkannya ke gambar baru
def train_and_cluster_images(images, k):
    all_flattened_pixels = []

    # Ubah ukuran gambar, ekstrak fitur, dan siapkan data
    for image in images:
        img = resize_image(image, target_size=(256, 256))  # Ubah ukuran gambar
        img = img.convert('RGB')  # Pastikan dalam format RGB
        img_data = np.array(img)
        img_data = normalize_pixels(img_data)  # Normalisasi nilai piksel menjadi [0, 1]
        h, w, _ = img_data.shape  # Pastikan gambar memiliki 3 channel (RGB)

        flattened_pixels = img_data.reshape(h * w, 3)  # Ubah piksel gambar menjadi bentuk (N, 3)
        all_flattened_pixels.append(flattened_pixels)

    data = np.vstack(all_flattened_pixels)  # Gabungkan semua data piksel yang diubah menjadi satu untuk pelatihan
    
    # Latih K-Means pada seluruh dataset
    clusters, centroids = kmeans_clustering(data, k)

    return clusters, centroids, data

# Aplikasi Streamlit
def main():
    st.title("Clustering Gambar dengan K-Means")

    # Unggah gambar
    uploaded_files = st.file_uploader("Pilih gambar...", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        # Pilih jumlah cluster
        k = st.slider("Pilih jumlah cluster", min_value=2, max_value=5, value=3)

        # Muat gambar
        images = [Image.open(uploaded_file) for uploaded_file in uploaded_files]

        # Tampilkan gambar asli
        st.subheader("Gambar Asli")
        for image in images:
            st.image(image, caption="Gambar Asli", use_column_width=True)

        # Proses semua gambar yang diunggah bersama
        if st.button("Cluster Gambar"):
            st.write("Clustering sedang berlangsung...")

            # Latih K-Means dan dapatkan gambar yang dikelompokkan
            clusters, centroids, data = train_and_cluster_images(images, k)

            # Tampilkan gambar yang dikelompokkan
            clustered_images = []
            for image in images:
                clustered_image = cluster_image_with_trained_model(image, centroids, resize_to=(256, 256))
                clustered_images.append(clustered_image)
                st.image(clustered_image, caption="Gambar Setelah Cluster", use_column_width=True)

            # Hitung nilai silhouette dengan metode lebih cepat (pengambilan sampel 10%)
            with st.spinner("Menghitung Skor Silhouette..."):
                silhouette_avg = calculate_silhouette_score_fast(data, clusters)

            # Tampilkan skor silhouette
            if silhouette_avg:
                st.write(f"Skor Silhouette untuk gambar : {silhouette_avg:.2f}")
            else:
                st.write("Skor Silhouette tidak dapat dihitung karena cluster yang tidak mencukupi.")

if __name__ == "__main__":
    main()

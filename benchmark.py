# import required libraries
import os, sys
import time
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.imagepro import (
    load, save,
    display_w_matplotlib,
    plot_centroids,
    get_pixels,
    plot_pixels_rgb_2d,
    plot_pixels_rgb_3d,
    calculate_psnr,
    calculate_ssim,
    calculate_compression_ratio
)
from kmeans.img_compression_kmeans_w_sklearn import apply_kmeans

# load original image
original_image_path = "images/original.jpg"
original_image_name, original_image_file_ext = original_image_path.split("/")[-1].split(".")

original_image = load(original_image_path)

# load cartoon image
cartoon_image_path = "images/cartoon.png"
cartoon_image_name, cartoon_image_file_ext = cartoon_image_path.split("/")[-1].split(".")

cartoon_image = load(cartoon_image_path)

# display both images side-by-side
# _, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(original_image)
# axs[0].set_title("Original Image")
# axs[1].imshow(cartoon_image)
# axs[1].set_title("Cartoon Image")
# # plt.show()

# apply kmeans to original image
time_start = time.time()
original_image_compressed, kmeans_original = apply_kmeans(original_image, 16)
original_image_compression_time = time.time() - time_start

# apply kmeans to cartoon image
time_start = time.time()
cartoon_image_compressed, kmeans_cartoon = apply_kmeans(cartoon_image, 16)
cartoon_image_compression_time = time.time() - time_start

# display compressed images side-by-side
# _, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(original_image_compressed)
# axs[0].set_title("Original Image Compressed")
# axs[1].imshow(cartoon_image_compressed)
# axs[1].set_title("Cartoon Image Compressed")
# plt.show()

# save compressed images
original_image_compressed_path = f"output/kmeans_sklearn_compressed_{original_image_name}.{original_image_file_ext}"
cartoon_image_compressed_path = f"output/kmeans_sklearn_compressed_{cartoon_image_name}.{cartoon_image_file_ext}"
save(original_image_compressed[:, :, ::-1], original_image_compressed_path)
save(cartoon_image_compressed[:, :, ::-1], cartoon_image_compressed_path)

# display centroids colors for original image
plot_centroids(kmeans_original, show=False, save_path="output/kmeans_sklearn_original_centroids.png")

# display centroids colors for cartoon image
plot_centroids(kmeans_cartoon, show=False, save_path="output/kmeans_sklearn_cartoon_centroids.png")

# display 2D pixel values for original image
plot_pixels_rgb_2d(get_pixels(original_image, optimized=True),
                   show=False, save_path="output/kmeans_sklearn_original_pixels_2d.png")

# display 2D pixel values for cartoon image
plot_pixels_rgb_2d(get_pixels(cartoon_image, optimized=True),
                   show=False, save_path="output/kmeans_sklearn_cartoon_pixels_2d.png")

# display 3D pixel values for original image
plot_pixels_rgb_3d(get_pixels(original_image, optimized=True),
                   show=False, save_path="output/kmeans_sklearn_original_pixels_3d.png")

# display 3D pixel values for cartoon image
plot_pixels_rgb_3d(get_pixels(cartoon_image, optimized=True),
                   show=False, save_path="output/kmeans_sklearn_cartoon_pixels_3d.png")

# Print Original image Compression Time
print(f"Original Image Compression Time:  {original_image_compression_time:.2f} seconds")

# Print Evaluation Metrics for original image
original_image_psnr_value = calculate_psnr(original_image, original_image_compressed)
print(f"Original Image PSNR:              {original_image_psnr_value:.2f} dB")

original_image_ssim_value = calculate_ssim(original_image, original_image_compressed)
print(f"Original Image SSIM:              {original_image_ssim_value:.4f}")

original_image_compression_ratio = calculate_compression_ratio(original_image_path, original_image_compressed_path)
print(f"Original Image Compression Ratio: {original_image_compression_ratio:.2f}x")

print(f"\n{'-'*50}\n")

# Print Cartoon image Compression Time
print(f"Cartoon Image Compression Time:   {cartoon_image_compression_time:.2f} seconds")

# Print Evaluation Metrics for cartoon image
cartoon_image_psnr_value = calculate_psnr(cartoon_image, cartoon_image_compressed)
print(f"Cartoon Image PSNR:               {cartoon_image_psnr_value:.2f} dB")

cartoon_image_ssim_value = calculate_ssim(cartoon_image, cartoon_image_compressed)
print(f"Cartoon Image SSIM:               {cartoon_image_ssim_value:.4f}")

cartoon_image_compression_ratio = calculate_compression_ratio(cartoon_image_path, cartoon_image_compressed_path)
print(f"Cartoon Image Compression Ratio:  {cartoon_image_compression_ratio:.2f}x")

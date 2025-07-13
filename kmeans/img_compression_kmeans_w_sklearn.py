# import required library
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim # Learn the details for "Structural Similarity Index (SSIM)"

# load image to RAM and convert it to a numpy array
def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = image[..., ::-1]
    return image_rgb

def display_image_matplotlib(image_rgb, save_path=None):
    plt.imshow(image_rgb)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def display_image_opencv(image):
    cv2.imshow('Image BGR', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_kmeans(image, n_clusters):
    # check if the image in a structure where we can process
    if image.shape[2] == 3:  # expected shape is (height, width, 3)
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

    else:
        print("Image is not in the expected format. Expected shape is (height, width, 3)")
        return None

def display_centroids(kmeans):
    # Create a visualization of the color palette (centroids)
    colors = kmeans.cluster_centers_.astype(np.uint8)  # Keep as 0-255 for display

    # Calculate grid dimensions dynamically based on number of colors
    grid_size = int(np.ceil(np.sqrt(kmeans.n_clusters)))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title(f'Color Palette ({kmeans.n_clusters} colors)')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Parameters for squares
    gap = 0.1  # space between squares
    size = 1 - gap  # size of each square

    # Plot each color as a square
    for i, color in enumerate(colors):
        row = i // grid_size
        col = i % grid_size

        # Calculate position (y is inverted in matplotlib)
        x = col + gap / 2
        y = grid_size - row - 1 + gap / 2

        # Normalized color for matplotlib
        rgb_norm = color / 255.0

        # Add colored rectangle
        rect = plt.Rectangle((x, y), size, size, color=rgb_norm)
        ax.add_patch(rect)

        # # Add RGB text
        # rgb_text = f"R:{color[0]}\nG:{color[1]}\nB:{color[2]}"
        # ax.text(x + size/2, y + size/2, rgb_text,
        #         ha='center', va='center', fontsize=8)

    # Add empty squares if needed to complete the grid
    for i in range(len(colors), grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        x = col + gap / 2
        y = grid_size - row - 1 + gap / 2
        rect = plt.Rectangle((x, y), size, size,
                             edgecolor='gray', facecolor='none', linestyle='dashed')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig('output/sklearn_kmeans_centroids.png', dpi=300)
    plt.show()

def get_image_pixels(image, optimized=True):
    pixels=image.reshape(-1, 3)
    pixels = pixels.astype(np.float32) / 255.0

    # Optional downsampling for performance
    if optimized and len(pixels) > 25000:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(pixels), size=25000, replace=False)
        pixels = pixels[indices]
    return pixels

def plot_pixels_rgb_3d(pixels):
    """
    Plot pixels of an image in 3D RGB space.

    Args:
        image (np.ndarray): Image array in shape (H, W, 3), RGB format, dtype uint8 or float32.
        sample_size (int): Maximum number of pixels to plot for performance.
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    ax.scatter(R, G, B, color=pixels, marker='.', s=2, alpha=0.6)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('Pixels in RGB Color Space')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    plt.tight_layout()
    plt.show()

def plot_pixels_rgb_2d(pixels):
    """
    Plot pixels of an image in 2D RGB space.

    Args:
        image (np.ndarray): Image array in shape (H, W, 3), RGB format, dtype uint8 or float32.
        sample_size (int): Maximum number of pixels to plot for performance.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    R, G, B = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    ax.scatter(R, G, color=pixels, marker='.', s=2, alpha=0.6)
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_title('Pixels in RGB Color Space')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('output/sklearn_kmeans_pixels_2d.png', dpi=300)
    plt.show()

def calculate_psnr(img1, img2):
    assert img1.shape == img2.shape, "Images must have the same dimensions"
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(img1, img2):
    """
    Calculate SSIM between two RGB images.
    Both images must be uint8 and have the same shape.
    """
    assert img1.shape == img2.shape, "Images must have the same dimensions"

    # SSIM expects grayscale or single-channel comparison per call
    ssim_total = 0.0
    for i in range(3):  # R, G, B
        channel_ssim = ssim(img1[..., i], img2[..., i], data_range=255)
        ssim_total += channel_ssim

    return ssim_total / 3.0

if __name__ == "__main__":
    image_path = "images/original.jpg"
    image = load_image(image_path)
    display_image_matplotlib(image)
    image_quantized, kmeans = apply_kmeans(image, 16)
    display_image_matplotlib(image_quantized, "output/sklearn_kmeans_quantized.png")

    # TODO: display centroids colors
    display_centroids(kmeans)

    # TODO: display each iteration of the model with centroids colors and color quantized image (side-by-side)
    # TODO: Display metrics like PSNR, SSIM, somewhere on the plot
    psnr_value = calculate_psnr(image, image_quantized)
    print(f"PSNR: {psnr_value:.2f} dB")

    ssim_val = calculate_ssim(image, image_quantized)
    print(f"SSIM: {ssim_val:.4f}")

    # TODO: display final version of the kmeans model clusters in 2D and 3D format plots
    pixels = get_image_pixels(image, optimized=True)
    plot_pixels_rgb_2d(pixels)
    plot_pixels_rgb_3d(pixels)

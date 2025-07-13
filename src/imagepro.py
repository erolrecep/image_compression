# import required libraries
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim # Learn the details for "Structural Similarity Index (SSIM)"


# load image to RAM and convert it to a numpy array
def load(image_path):
    image = cv2.imread(image_path)
    image_rgb = image[..., ::-1]
    return image_rgb

def save(image, save_path):
    cv2.imwrite(save_path, image)

def display_w_matplotlib(image_rgb, save_path=None):
    plt.imshow(image_rgb)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def display_w_opencv(image):
    cv2.imshow('Image BGR', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_centroids(kmeans, show=False, save_path=None):
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

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        # Close the plot to free up memory
        plt.close()

def get_pixels(image, optimized=True):
    pixels=image.reshape(-1, 3)
    pixels = pixels.astype(np.float32) / 255.0

    # Optional downsampling for performance
    if optimized and len(pixels) > 25000:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(pixels), size=25000, replace=False)
        pixels = pixels[indices]
    return pixels

def plot_pixels_rgb_3d(pixels, show=False, save_path=None):
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
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plot_pixels_rgb_2d(pixels, show=False, save_path=None, optimized=True):
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
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def calculate_psnr(original, compressed):
    assert original.shape == compressed.shape, "Images must have the same dimensions"
    mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def calculate_ssim(original, compressed):
    """
    Calculate SSIM between two RGB images.
    Both images must be uint8 and have the same shape.
    """
    assert original.shape == compressed.shape, "Images must have the same dimensions"

    # SSIM expects grayscale or single-channel comparison per call
    ssim_total = 0.0
    for i in range(3):  # R, G, B
        channel_ssim = ssim(original[..., i], compressed[..., i], data_range=255)
        ssim_total += channel_ssim

    return ssim_total / 3.0

def calculate_compression_ratio(original_image_path, compressed_image_path):
    # print(f"[INFO] Calculating compression ratio for {original_image_path} and {compressed_image_path}...]")
    try:
        # Get original image size
        original_size = os.path.getsize(original_image_path)

        # Get compressed image size
        compressed_size = os.path.getsize(compressed_image_path)

        # Calculate compression ratio
        if compressed_size > 0:
            compression_ratio = original_size / compressed_size
            return compression_ratio
        else:
            print(f"Error: Compressed image size is zero for {compressed_image_path}")
            return None

    except FileNotFoundError:
        print(f"Error: Image file not found at {original_image_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

import argparse
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Avoid potential Qt platform rendering issues


def detect_coastline_water_based(image_path):
    """
    Coastline extraction method based on water body detection

    This function performs the following steps:
    1. Read input image
    2. Convert image to RGB and HSV color spaces
    3. Create water body mask using color thresholding
    4. Apply morphological operations to refine mask
    5. Find and filter water body contours
    6. Detect coastline segments

    Args:
        image_path (str): Path to input image file

    Returns:
        tuple: Contains four elements
        - original RGB image
        - water body mask
        - coastline detection image
        - number of coastline segments
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Convert BGR to RGB color space
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space for water detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV range for water body detection (blue spectrum)
    lower_water = np.array([90, 40, 40])  # Blue lower threshold
    upper_water = np.array([130, 255, 255])  # Blue upper threshold

    # Create water body mask using color thresholding
    water_mask = cv2.inRange(hsv, lower_water, upper_water)

    # Morphological closing to fill internal water body holes
    kernel = np.ones((15, 15), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)

    # Remove small noise regions
    kernel_small = np.ones((5, 5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel_small)

    # Find water body contours
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small water regions, keep only main ocean areas
    main_water_mask = np.zeros_like(water_mask)
    min_area = img.shape[0] * img.shape[1] * 0.01  # Minimum area threshold (1% of image)

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.fillPoly(main_water_mask, [contour], 255)
            valid_contours.append(contour)

    # Create coastline detection visualization
    coastline_img = img_rgb.copy()
    coastline_contours = []

    for contour in valid_contours:
        # Simplify contour for smoother coastline representation
        epsilon = 0.001 * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(simplified_contour) > 10:
            coastline_contours.append(simplified_contour)
            cv2.drawContours(coastline_img, [simplified_contour], -1, (0, 255, 0), 3)

    return img_rgb, main_water_mask, coastline_img, len(coastline_contours)


def create_three_panel_plot(original_img, water_mask, coastline_img, coastline_count, base_name, save_path):
    """
    Create a three-panel visualization of coastline analysis results

    This function generates a figure with three subplots:
    1. Original image
    2. Water body segmentation mask
    3. Coastline detection result

    Args:
        original_img (ndarray): Original RGB image
        water_mask (ndarray): Water body segmentation mask
        coastline_img (ndarray): Coastline detection visualization
        coastline_count (int): Number of detected coastline segments
        base_name (str): Base filename for title
        save_path (str): Output file path for saving figure

    Returns:
        str: Path to saved visualization image
    """
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Set overall figure title
    fig.suptitle(f'Coastline Analysis - {base_name}', fontsize=16, fontweight='bold', y=0.95)

    # First subplot: Original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold', pad=20)
    axes[0].axis('off')

    # Second subplot: Water segmentation mask
    axes[1].imshow(water_mask, cmap='gray')
    axes[1].set_title('Water Segmentation', fontsize=14, fontweight='bold', pad=20)
    axes[1].axis('off')

    # Third subplot: Coastline detection
    axes[2].imshow(coastline_img)
    axes[2].set_title(f'Coastline Detection', fontsize=14, fontweight='bold', pad=20)
    axes[2].axis('off')

    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.02, right=0.98, wspace=0.05)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close figure to release memory

    return save_path


def advanced_coastline_detection(image_path):
    """
    Advanced coastline detection method using multiple color space techniques

    This function implements a multi-stage water body and coastline detection process:
    1. Convert image to different color spaces (RGB, HSV, LAB)
    2. Create multiple water body masks using different color thresholding methods
    3. Combine and refine masks using morphological operations
    4. Detect and simplify coastline contours

    Args:
        image_path (str): Path to the input image file

    Returns:
        tuple: Containing four elements
        - Original RGB image
        - Main water body mask
        - Coastline visualization image
        - Number of detected coastline segments
    """
    # Read input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Convert image to RGB color space
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Method 1: Water detection using HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_water1 = np.array([95, 30, 30])  # Lower HSV threshold for water
    upper_water1 = np.array([125, 255, 255])  # Upper HSV threshold for water
    mask1 = cv2.inRange(hsv, lower_water1, upper_water1)

    # Method 2: Water detection using LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mask2 = cv2.inRange(b, 0, 115)  # Use B channel for water detection

    # Method 3: Blue color detection in RGB space
    mask3 = cv2.inRange(img_rgb, np.array([0, 50, 100]), np.array([100, 150, 255]))

    # Combine multiple masks using bitwise OR operation
    combined_mask = cv2.bitwise_or(mask1, mask2)
    combined_mask = cv2.bitwise_or(combined_mask, mask3)

    # Morphological processing to refine water body mask
    kernel = np.ones((20, 20), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))  # Remove small noise

    # Find water body contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize main water mask and coastline visualization
    main_water_mask = np.zeros_like(combined_mask)
    min_area = img.shape[0] * img.shape[1] * 0.005  # Minimum area threshold (0.5% of image)

    coastline_contours = []
    coastline_img = img_rgb.copy()

    # Process detected contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Fill valid water region in main mask
            cv2.fillPoly(main_water_mask, [contour], 255)

            # Simplify contour for smoother coastline representation
            epsilon = 0.0015 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)

            # Add valid coastline segments
            if len(simplified) > 15:
                coastline_contours.append(simplified)
                cv2.drawContours(coastline_img, [simplified], -1, (0, 255, 0), 2)

    return img_rgb, main_water_mask, coastline_img, len(coastline_contours)


def create_detailed_analysis_plot(original_img, water_mask, coastline_img, coastline_count, base_name, save_path):
    """
    Create a comprehensive 2x2 grid visualization of coastline analysis results

    This function generates a detailed figure with four subplots:
    1. Original image
    2. Water segmentation mask (blue colormap)
    3. Coastline detection result
    4. Overlay of original image with semi-transparent water mask

    Args:
        original_img (ndarray): Original RGB image
        water_mask (ndarray): Water body segmentation mask
        coastline_img (ndarray): Coastline detection visualization
        coastline_count (int): Number of detected coastline segments
        base_name (str): Base filename for plot title
        save_path (str): Output file path for saving visualization

    Returns:
        str: Path to saved detailed analysis image
    """
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Set overall figure title
    fig.suptitle(f'Detailed Coastline Analysis - {base_name}', fontsize=18, fontweight='bold', y=0.95)

    # First subplot: Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Second subplot: Water segmentation mask
    axes[0, 1].imshow(water_mask, cmap='Blues')
    axes[0, 1].set_title('Water Segmentation', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # Third subplot: Coastline detection
    axes[1, 0].imshow(coastline_img)
    axes[1, 0].set_title(f'Coastline Detection ({coastline_count} segments)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Fourth subplot: Overlay view
    overlay_img = original_img.copy()
    # Create semi-transparent water body overlay
    water_overlay = np.zeros_like(original_img)
    water_overlay[water_mask > 0] = [0, 100, 255]  # Blue color for water overlay
    overlay_result = cv2.addWeighted(overlay_img, 0.7, water_overlay, 0.3, 0)

    axes[1, 1].imshow(overlay_result)
    axes[1, 1].set_title('Overlay View', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Adjust subplot layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.02, right=0.98, wspace=0.05, hspace=0.15)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close figure to release memory

    return save_path


def process_single_image(image_path, mode, output_dir):
    """
    process single image
    """
    try:
        print(f"Processing: {os.path.basename(image_path)}")

        if mode == 'advanced':
            original_img, water_mask, coastline_img, coastline_count = advanced_coastline_detection(image_path)
            mode_suffix = '_advanced'
        else:
            original_img, water_mask, coastline_img, coastline_count = detect_coastline_water_based(image_path)
            mode_suffix = '_basic'

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        # output images
        result_path = os.path.join(output_dir, f'{base_name}_{mode_suffix}.jpg')
        create_three_panel_plot(original_img, water_mask, coastline_img, coastline_count, base_name, result_path)

        return True
    except Exception as e:
        print(f"Error when processing {image_path}:  {e}")
        return False


def process_batch(input_dir, mode, output_dir):
    """
    batch process
        input_dir = folder path
        mode = basic or advanced
    """
    if not os.path.exists(input_dir):
        print(f"{input_dir} doesn't exist, please check!")
        return

    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']
    processed = 0

    for filename in sorted(os.listdir(input_dir)):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_path = os.path.join(input_dir, filename)

            if process_single_image(image_path, mode, output_dir):
                processed += 1

    print(f"\nFinished {processed} images.")


def parse_args():
    # parser = argparse.ArgumentParser(description='Coastline Detection Tool',
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    # Input dir
    parser.add_argument("--input_path", type=str, help="Input file path.",
                        default="./sa_data")

    # Dir or single image
    parser.add_argument("--images", type=str, help="Image or Directory",
                        default="d")

    # Running mode: Basic or Advanced
    parser.add_argument("--mode", type=str, help="Running mode: basic or advanced.")

    # Output dir
    parser.add_argument("--output-path", type=str, help="Output file path.",
                        default="./result")

    return parser.parse_args()


def main(opt):
    if opt.images == 'd':
        input_dir = opt.input_path
        output_dir = opt.output_path
        mode = opt.mode

        process_batch(input_dir, mode, output_dir)
    else:
        input_dir = opt.input_path
        output_dir = opt.output_path
        mode = opt.mode

        process_single_image(input_dir, mode, output_dir)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
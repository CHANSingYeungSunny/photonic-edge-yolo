import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    # Load input image (cv2 loads as BGR)
    input_path = 'input.jpg'
    original_bgr = cv2.imread(input_path)
    
    # Image Preprocessing for Edge Detection
    # 1. Convert to grayscale
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Sobel filter
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. Combine gradients using magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 4. Normalize to 0-255 range
    sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_normalized = np.uint8(sobel_normalized)
    
    # 5. Invert colors (white edges on black background)
    edge_image = cv2.bitwise_not(sobel_normalized)
    
    # 6. Convert edge image to 3-channel RGB for YOLO input
    edge_rgb = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2RGB)
    
    # Load YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Run inference on original image (convert to RGB for YOLO)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    results_original = model(original_rgb)
    
    # Run inference on edge-only image
    results_edge = model(edge_rgb)
    
    # Visualization Setup
    plt.figure(figsize=(20, 10), dpi=300)  # High-res figure
    
    # Plot 1: Original Image with Bounding Boxes
    plt.subplot(1, 2, 1)
    # Draw bounding boxes on original image
    annotated_original = results_original[0].plot(line_width=2)
    plt.imshow(annotated_original)
    plt.title('Standard GPU (RGB)', fontsize=16)
    plt.axis('off')
    
    # Plot 2: Edge-Only Image with Bounding Boxes
    plt.subplot(1, 2, 2)
    # Draw bounding boxes on edge image
    annotated_edge = results_edge[0].plot(line_width=2)
    plt.imshow(annotated_edge)
    plt.title('Photonic Simulation (Edge-Only)', fontsize=16)
    plt.axis('off')
    
    # Add main title
    plt.suptitle('Photonic Edge Detection vs Standard RGB YOLOv8', fontsize=20, y=0.95)
    
    # Save and display
    plt.tight_layout()
    plt.savefig('comparison_benchmark.png', dpi=300, bbox_inches='tight')
    print("Comparison benchmark saved as 'comparison_benchmark.png'")

if __name__ == '__main__':
    main()
## Implementation Plan

### Script Structure
1. **Import Libraries**: cv2, numpy, matplotlib, ultralytics
2. **Load Input Image**: Read `input.jpg` using cv2
3. **Image Preprocessing for Edge Detection**:
   - Convert to grayscale
   - Apply Sobel filter (kernel size 3) for X and Y gradients
   - Combine gradients using magnitude calculation
   - Normalize to 0-255 range
   - Invert colors (white edges on black background)
   - Convert edge image to 3-channel RGB (required for YOLO input)
4. **YOLOv8 Inference**:
   - Load YOLOv8n model from ultralytics
   - Run inference on original RGB image
   - Run inference on edge-only RGB image
5. **Visualization**:
   - Create matplotlib figure with 2 subplots
   - Left: Original image with bounding boxes (convert BGR to RGB for display)
   - Right: Edge-only image with bounding boxes
   - Add titles: "Standard GPU (RGB)" vs "Photonic Simulation (Edge-Only)"
   - Save as `comparison_benchmark.png`

### Key Considerations
- YOLOv8 expects 3-channel RGB input, so edge image will be converted from grayscale to 3-channel
- Proper color space conversion between cv2 (BGR) and matplotlib (RGB)
- Sobel filter parameters: kernel size 3, proper gradient combination
- High-resolution output for research benchmark purposes

### Files to Create
- `photonic_edge_detection.py`: Main script implementing the pipeline
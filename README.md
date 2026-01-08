# Photonic-Edge-YOLO: Benchmarking Analog Differentiation for Edge AI

## Context
This project benchmarks the viability of using high-speed photonic chips (simulated via Sobel differentiation) as a pre-processing frontend for YOLOv8 object detection. It explores whether edge detection can serve as an effective bandwidth reduction technique while maintaining detection accuracy for autonomous systems.

## Methodology
To validate the robustness of this pipeline for autonomous navigation, we utilized sample imagery from the KITTI Vision Benchmark Suite. The KITTI dataset provides realistic urban driving scenarios, making it an ideal testbed for evaluating object detection performance under simulated photonic pre-processing.

## Input
Standard RGB frames from the KITTI dataset, starting with a sample `input.jpg` included in this repository.

## Simulation Pipeline
1. **Grayscale Conversion**: Convert RGB input to grayscale to reduce dimensionality
2. **Sobel Gradient Filter**: Apply hardware-accurate Sobel filters in both X and Y directions to compute image gradients
3. **Gradient Combination**: Calculate gradient magnitude to form edge map
4. **Normalization**: Scale gradient values to 0-255 range
5. **Color Inversion**: Produce high-contrast edge image with white edges on black background, simulating photonic chip output

## Inference
We run the YOLOv8n (nano) model on both the original RGB images and the edge-only images to:
- Compare detection accuracy between standard and edge-only inputs
- Test the model's robustness to extreme bandwidth reduction
- Evaluate the viability of photonic pre-processing for real-time applications

## Results
![Result](comparison_benchmark.png)

The comparison demonstrates that YOLOv8 can still detect objects effectively even when fed only edge information, retaining structural integrity for reliable detection despite significant bandwidth reduction.

## Usage
1. Ensure you have the required dependencies installed:
   ```bash
   pip install opencv-python numpy matplotlib ultralytics
   ```

2. Run the main simulation script:
   ```bash
   python photonic_edge_detection.py
   ```

3. The script will:
   - Load `input.jpg` from the repository
   - Generate the edge-only image using Sobel filtering
   - Run YOLOv8n inference on both original and edge images
   - Create and save the comparison benchmark as `comparison_benchmark.png`

## Requirements
- Python 3.8+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Ultralytics (for YOLOv8)

## Project Structure
```
Photonic-Edge-YOLO/
├── photonic_edge_detection.py  # Main simulation script
├── input.jpg                   # Sample KITTI image
├── comparison_benchmark.png    # Generated comparison result
├── yolov8n.pt                  # YOLOv8 nano model (downloaded automatically)
└── README.md                   # This file
```

## Conclusion
This benchmark demonstrates that photonic edge detection could be a viable approach for high-speed, low-bandwidth object detection pipelines. By reducing the input data to only edge information while maintaining detection accuracy, this approach shows promise for next-generation autonomous systems utilizing photonic processing engines.
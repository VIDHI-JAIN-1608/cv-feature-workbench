# Interactive Edge & Feature Detection Workbench

## Overview
A lightweight Computer Vision workspace applying basic preprocessing, edge detection, and geometric feature extraction using OpenCV.

## Features
- **Enhancement Module**: Contrast normalization via Histogram Equalization.
- **Segmentation Module**: Canny Edge detection combined with Morphological operations.
- **Feature Extraction Module**: Harris Corner and Hough Line detection.

## Technologies Used
- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Installation & Setup
1. Clone repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run pipeline: `python main.py sample.jpg`

## Testing Instructions
Run `python main.py sample.jpg` with any sample image to verify generation of `output_enhanced.png`, `output_edges.png`, and `output_features.png`.

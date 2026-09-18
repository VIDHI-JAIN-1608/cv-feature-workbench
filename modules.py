import cv2
import numpy as np

def enhance_image(image):
    """Module 1: Preprocessing & Contrast Enhancement"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized

def process_edges_and_morphology(gray_img, low_thresh=50, high_thresh=150):
    """Module 2: Edge Detection and Morphological Operations"""
    edges = cv2.Canny(gray_img, low_thresh, high_thresh)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    return dilated_edges

def detect_features(gray_img, original_img):
    """Module 3: Corner (Harris) and Line (Hough) Detection"""
    output = original_img.copy()
    
    # 1. Harris Corner Detection
    dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    output[dst > 0.01 * dst.max()] = [0, 0, 255] # Red dots for corners
    
    # 2. Hough Line Transform
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green lines
            
    return output

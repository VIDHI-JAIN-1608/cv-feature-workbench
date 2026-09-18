import cv2
import sys
from modules import enhance_image, process_edges_and_morphology, detect_features

def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image!")
        return

    # Pipeline execution
    enhanced = enhance_image(img)
    edges = process_edges_and_morphology(enhanced)
    features = detect_features(enhanced, img)

    # Save outputs for submission report
    cv2.imwrite("output_enhanced.png", enhanced)
    cv2.imwrite("output_edges.png", edges)
    cv2.imwrite("output_features.png", features)
    print("Processing complete. Output files generated successfully.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    main(path)

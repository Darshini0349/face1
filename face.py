"""
Project 1 (CV): Face Mask Detection
A two-stage pipeline: (1) detect faces in an image with OpenCV's real,
bundled Haar cascade, then (2) decide mask / no-mask for each face.

HOW THE TWO STAGES WORK:
  Stage 1 (face detection) is REAL and production-usable — it uses the
  Haar cascade that ships with OpenCV, no download or training needed.
  Stage 2 (mask classification) is where a real project plugs in a trained
  CNN (MobileNetV2 transfer learning is the standard choice — which you've
  already done for plant disease). Since training that needs a labelled
  mask dataset we can't download here, this script uses a transparent
  HEURISTIC (checks whether the lower-face region is skin-toned vs covered)
  so the full pipeline runs end to end. The classify_mask() function is
  clearly marked as the swap point for a real model.

Run: pip install opencv-python numpy
     python 01_face_mask_detection.py path/to/image.jpg
     (with no argument, it runs on a generated synthetic test image)

Concepts practiced: cascade-based object detection, region-of-interest
extraction, the detect-then-classify pattern used across CV, and honest
separation of "real component" vs "swap-in-a-model component".
"""

import sys
import cv2
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(gray_image):
    """Stage 1: REAL face detection via Haar cascade."""
    return FACE_CASCADE.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))


def classify_mask(face_bgr):
    """
    Stage 2: mask vs no-mask.

    >>> SWAP POINT <<<  In a real project, replace this whole function with:
        prediction = cnn_model.predict(preprocess(face_bgr))
        return "Mask" if prediction[0] > 0.5 else "No Mask", confidence

    Demo heuristic: a mask covers the lower half of the face. If that region
    is NOT skin-toned (masks are usually blue/white/black), we guess "Mask".
    This is a stand-in for a trained classifier, not a real one.
    """
    h, w = face_bgr.shape[:2]
    lower_face = face_bgr[h // 2 : h, :]  # bottom half = mouth/chin area

    hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
    # Rough skin-tone range in HSV.
    skin_lower = np.array([0, 30, 60])
    skin_upper = np.array([25, 170, 255])
    skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)

    skin_ratio = np.count_nonzero(skin_mask) / skin_mask.size
    # Lots of skin visible in the lower face -> probably NO mask.
    if skin_ratio > 0.30:
        return "No Mask", 1 - skin_ratio
    return "Mask", 1 - skin_ratio


def make_synthetic_test_image():
    """A simple face-like image so the script runs without any input file."""
    img = np.full((300, 300, 3), 200, dtype=np.uint8)
    # skin-toned oval "face"
    cv2.ellipse(img, (150, 150), (80, 100), 0, 0, 360, (180, 150, 130), -1)
    # eyes
    cv2.circle(img, (120, 120), 10, (50, 50, 50), -1)
    cv2.circle(img, (180, 120), 10, (50, 50, 50), -1)
    return img


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    print(f"Faces detected: {len(faces)}")
    if len(faces) == 0:
        print("(Haar cascades work best on clear, front-facing real photos.)")
        return

    for i, (x, y, w, h) in enumerate(faces, start=1):
        face_region = img[y : y + h, x : x + w]
        label, confidence = classify_mask(face_region)
        print(f"  Face {i} at ({x},{y}) size {w}x{h}: {label} (score {confidence:.2f})")


def main():
    print("=== Face Mask Detection (Haar detection + classifier stage) ===")
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Could not read image: {sys.argv[1]}")
            return
        print(f"Loaded {sys.argv[1]}")
    else:
        print("No image provided — using a synthetic test image.")
        img = make_synthetic_test_image()

    process_image(img)


if __name__ == "__main__":
    main()

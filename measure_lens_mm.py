import cv2
import numpy as np
import os


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = r"C:\Users\luisf\PycharmProjects\optiframe\input\lens01.webp"

MASK_PATH = "output/lens_mask_clean.webp"

# If you haven't created the cleaned mask yet,
# it will automatically try the original SAM mask.
RAW_MASK_PATH = "output/lens_mask.webp"

MARKER_SIZE_MM = 37.2


# ============================================================
# 1. LOAD ORIGINAL IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError("Could not load original image.")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# 2. DETECT ARUCO MARKERS
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)

corners, ids, rejected = detector.detectMarkers(gray)

if ids is None or len(ids) < 4:
    raise RuntimeError(
        "We need all 4 ArUco markers."
    )

print("Markers detected:", ids.flatten())


# ============================================================
# 3. FIND MARKER CENTERS
# ============================================================

marker_centers = []

for marker in corners:

    points = marker.reshape(4, 2)

    center = np.mean(
        points,
        axis=0
    )

    marker_centers.append(center)


marker_centers = np.array(
    marker_centers,
    dtype=np.float32
)


# ============================================================
# 4. ORDER THE 4 MARKER CENTERS
# ============================================================

ordered = np.zeros(
    (4, 2),
    dtype=np.float32
)

sums = marker_centers.sum(axis=1)

differences = np.diff(
    marker_centers,
    axis=1
).flatten()


ordered[0] = marker_centers[
    np.argmin(sums)
]

ordered[2] = marker_centers[
    np.argmax(sums)
]

ordered[1] = marker_centers[
    np.argmin(differences)
]

ordered[3] = marker_centers[
    np.argmax(differences)
]


# ============================================================
# 5. SAME RECTIFICATION SIZE AS main.py
# ============================================================

top_left = ordered[0]
top_right = ordered[1]
bottom_right = ordered[2]
bottom_left = ordered[3]


top_width = np.linalg.norm(
    top_right - top_left
)

bottom_width = np.linalg.norm(
    bottom_right - bottom_left
)

left_height = np.linalg.norm(
    bottom_left - top_left
)

right_height = np.linalg.norm(
    bottom_right - top_right
)


output_width = int(
    max(top_width, bottom_width)
)

output_height = int(
    max(left_height, right_height)
)


destination = np.array(
    [
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ],
    dtype=np.float32
)


# ============================================================
# 6. CALCULATE SAME HOMOGRAPHY
# ============================================================

matrix = cv2.getPerspectiveTransform(
    ordered,
    destination
)


# ============================================================
# 7. TRANSFORM THE ACTUAL ARUCO CORNERS
# ============================================================

horizontal_marker_edges = []
vertical_marker_edges = []


for marker in corners:

    original_points = marker.reshape(
        1,
        4,
        2
    ).astype(np.float32)

    transformed = cv2.perspectiveTransform(
        original_points,
        matrix
    )[0]

    # Check all four edges of the marker
    for i in range(4):

        p1 = transformed[i]
        p2 = transformed[(i + 1) % 4]

        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])

        length = np.linalg.norm(
            p2 - p1
        )

        # Determine whether this marker edge
        # is approximately horizontal or vertical
        if dx >= dy:

            horizontal_marker_edges.append(
                length
            )

        else:

            vertical_marker_edges.append(
                length
            )


# ============================================================
# 8. CALCULATE PIXELS PER MILLIMETER
# ============================================================

average_marker_width_px = np.mean(
    horizontal_marker_edges
)

average_marker_height_px = np.mean(
    vertical_marker_edges
)


pixels_per_mm_x = (
    average_marker_width_px
    / MARKER_SIZE_MM
)

pixels_per_mm_y = (
    average_marker_height_px
    / MARKER_SIZE_MM
)


print()
print("===== CALIBRATION =====")
print()

print(
    "Marker real size:",
    MARKER_SIZE_MM,
    "mm"
)

print(
    "Average marker width:",
    round(average_marker_width_px, 2),
    "pixels"
)

print(
    "Average marker height:",
    round(average_marker_height_px, 2),
    "pixels"
)

print(
    "Horizontal scale:",
    round(pixels_per_mm_x, 4),
    "pixels/mm"
)

print(
    "Vertical scale:",
    round(pixels_per_mm_y, 4),
    "pixels/mm"
)


# ============================================================
# 9. LOAD SEGMENTATION MASK
# ============================================================

if os.path.exists(MASK_PATH):

    mask = cv2.imread(
        MASK_PATH,
        cv2.IMREAD_GRAYSCALE
    )

else:

    print(
        "\nClean mask not found."
        " Using original SAM mask."
    )

    mask = cv2.imread(
        RAW_MASK_PATH,
        cv2.IMREAD_GRAYSCALE
    )


if mask is None:

    raise FileNotFoundError(
        "Could not load segmentation mask."
    )


# ============================================================
# 10. CLEAN MASK
# ============================================================

_, binary_mask = cv2.threshold(
    mask,
    127,
    255,
    cv2.THRESH_BINARY
)


contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


if not contours:

    raise RuntimeError(
        "No lens found in segmentation mask."
    )


largest_contour = max(
    contours,
    key=cv2.contourArea
)


# ============================================================
# 11. FIND LENS WIDTH AND HEIGHT IN PIXELS
# ============================================================

x, y, width_px, height_px = cv2.boundingRect(
    largest_contour
)


# ============================================================
# 12. CONVERT PIXELS TO MILLIMETERS
# ============================================================

width_mm = (
    width_px
    / pixels_per_mm_x
)

height_mm = (
    height_px
    / pixels_per_mm_y
)


area_pixels = cv2.contourArea(
    largest_contour
)

area_mm2 = (
    area_pixels
    /
    (
        pixels_per_mm_x
        *
        pixels_per_mm_y
    )
)


# ============================================================
# 13. DISPLAY FINAL RESULTS
# ============================================================

print()
print("==============================")
print("      LENS MEASUREMENTS")
print("==============================")
print()

print(
    "Width in pixels:",
    width_px
)

print(
    "Height in pixels:",
    height_px
)

print()

print(
    "Lens width:",
    round(width_mm, 2),
    "mm"
)

print(
    "Lens height:",
    round(height_mm, 2),
    "mm"
)

print(
    "Lens area:",
    round(area_mm2, 2),
    "mm²"
)

print()
print("==============================")
print("SUCCESS")
print("==============================")
import cv2
import numpy as np
import os


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = r"C:\Users\luisf\PycharmProjects\optiframe\input\lens01.jpg"

OUTPUT_ARUCO = "output/aruco_test.jpg"
OUTPUT_RECTIFIED = "output/rectified.jpg"
OUTPUT_SCALE = "output/scale.txt"

MARKER_SIZE_MM = 37.2


# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not open image.")
    exit()

print("Image loaded successfully!")


# ============================================================
# 2. DETECT ARUCO MARKERS
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)

corners, ids, rejected = detector.detectMarkers(gray)

if ids is None:
    print("ERROR: No ArUco markers detected.")
    exit()

print("Number of markers detected:", len(ids))
print("Marker IDs:", ids.flatten())

if len(ids) < 4:
    print("ERROR: We need all 4 ArUco markers.")
    exit()


# ============================================================
# 3. DRAW DETECTED MARKERS
# ============================================================

aruco_result = image.copy()

cv2.aruco.drawDetectedMarkers(
    aruco_result,
    corners,
    ids
)

os.makedirs(
    "output",
    exist_ok=True
)

cv2.imwrite(
    OUTPUT_ARUCO,
    aruco_result
)


# ============================================================
# 4. FIND CENTER OF EACH MARKER
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
# 5. ORDER MARKER CENTERS
#
# 0 = TOP LEFT
# 1 = TOP RIGHT
# 2 = BOTTOM RIGHT
# 3 = BOTTOM LEFT
# ============================================================

ordered = np.zeros(
    (4, 2),
    dtype=np.float32
)

sums = marker_centers.sum(
    axis=1
)

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


print("\nMarker center positions:")

print(
    "Top-left:",
    ordered[0]
)

print(
    "Top-right:",
    ordered[1]
)

print(
    "Bottom-right:",
    ordered[2]
)

print(
    "Bottom-left:",
    ordered[3]
)


# ============================================================
# 6. CALCULATE OUTPUT SIZE
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
    max(
        top_width,
        bottom_width
    )
)

output_height = int(
    max(
        left_height,
        right_height
    )
)


print(
    "\nRectified size:",
    output_width,
    "x",
    output_height,
    "pixels"
)


# ============================================================
# 7. DESTINATION RECTANGLE
# ============================================================

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
# 8. PERSPECTIVE TRANSFORM
# ============================================================

matrix = cv2.getPerspectiveTransform(
    ordered,
    destination
)


# ============================================================
# 9. CREATE RECTIFIED IMAGE
# ============================================================

rectified = cv2.warpPerspective(
    image,
    matrix,
    (
        output_width,
        output_height
    )
)


cv2.imwrite(
    OUTPUT_RECTIFIED,
    rectified
)


# ============================================================
# 10. TRANSFORM ARUCO CORNERS
# ============================================================

horizontal_sides = []
vertical_sides = []


for marker in corners:

    points = marker.reshape(
        4,
        2
    ).astype(np.float32)

    transformed_points = cv2.perspectiveTransform(
        points.reshape(1, -1, 2),
        matrix
    )[0]


    # ArUco corner order:
    #
    # 0 = top-left
    # 1 = top-right
    # 2 = bottom-right
    # 3 = bottom-left


    # ----------------------------------------
    # Horizontal marker sides
    # ----------------------------------------

    top_side = np.linalg.norm(
        transformed_points[1]
        -
        transformed_points[0]
    )

    bottom_side = np.linalg.norm(
        transformed_points[2]
        -
        transformed_points[3]
    )


    # ----------------------------------------
    # Vertical marker sides
    # ----------------------------------------

    right_side = np.linalg.norm(
        transformed_points[2]
        -
        transformed_points[1]
    )

    left_side = np.linalg.norm(
        transformed_points[3]
        -
        transformed_points[0]
    )


    horizontal_sides.extend(
        [
            top_side,
            bottom_side
        ]
    )

    vertical_sides.extend(
        [
            left_side,
            right_side
        ]
    )


# ============================================================
# 11. CALCULATE X AND Y PIXEL SCALES
# ============================================================

average_horizontal_pixels = np.mean(
    horizontal_sides
)

average_vertical_pixels = np.mean(
    vertical_sides
)


pixels_per_mm_x = (
    average_horizontal_pixels
    /
    MARKER_SIZE_MM
)

pixels_per_mm_y = (
    average_vertical_pixels
    /
    MARKER_SIZE_MM
)


mm_per_pixel_x = (
    1.0
    /
    pixels_per_mm_x
)

mm_per_pixel_y = (
    1.0
    /
    pixels_per_mm_y
)


# ============================================================
# 12. PRINT CALIBRATION RESULTS
# ============================================================

print("\n===== CALIBRATION =====")

print(
    "Average horizontal marker size:",
    round(
        average_horizontal_pixels,
        2
    ),
    "pixels"
)

print(
    "Average vertical marker size:",
    round(
        average_vertical_pixels,
        2
    ),
    "pixels"
)


print()

print(
    "X pixels per mm:",
    round(
        pixels_per_mm_x,
        4
    )
)

print(
    "Y pixels per mm:",
    round(
        pixels_per_mm_y,
        4
    )
)


print()

print(
    "X mm per pixel:",
    round(
        mm_per_pixel_x,
        4
    )
)

print(
    "Y mm per pixel:",
    round(
        mm_per_pixel_y,
        4
    )
)


# ============================================================
# 13. SAVE CALIBRATION
# ============================================================

with open(
    OUTPUT_SCALE,
    "w"
) as file:

    file.write(
        f"{pixels_per_mm_x}\n"
    )

    file.write(
        f"{pixels_per_mm_y}\n"
    )


# ============================================================
# 14. FINISHED
# ============================================================

print("\nSUCCESS")

print(
    "ArUco image saved to:",
    OUTPUT_ARUCO
)

print(
    "Rectified image saved to:",
    OUTPUT_RECTIFIED
)

print(
    "Scale saved to:",
    OUTPUT_SCALE
)

print(
    "Marker physical size:",
    MARKER_SIZE_MM,
    "mm"
)
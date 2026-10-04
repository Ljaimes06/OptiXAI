import cv2
import numpy as np


# ============================================================
# FILES
# ============================================================

IMAGE_PATH = "output/rectified.jpg"
MASK_PATH = "output/lens_mask.jpg"

OUTPUT_CLEAN_MASK = "output/lens_mask_clean.jpg"
OUTPUT_ISOLATED = "output/isolated_lens.jpg"
OUTPUT_MEASURED = "output/lens_measured.jpg"

# ============================================================
# 1. LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        "Could not load output/rectified.webp"
    )


# ============================================================
# 2. LOAD SAM MASK
# ============================================================

mask = cv2.imread(
    MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "Could not load output/lens_mask.webp"
    )


# ============================================================
# 3. LOAD X AND Y CALIBRATION
# ============================================================

with open(SCALE_PATH, "r") as file:
    values = file.readlines()

if len(values) < 2:
    raise RuntimeError(
        "scale.txt must contain X and Y pixels/mm."
    )

pixels_per_mm_x = float(values[0].strip())
pixels_per_mm_y = float(values[1].strip())

print("Calibration loaded:")
print("X pixels/mm:", round(pixels_per_mm_x, 4))
print("Y pixels/mm:", round(pixels_per_mm_y, 4))


# ============================================================
# 4. CREATE BINARY MASK
# ============================================================

_, binary_mask = cv2.threshold(
    mask,
    127,
    255,
    cv2.THRESH_BINARY
)


# ============================================================
# 5. FIND SEGMENTED OBJECT
# ============================================================

contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise RuntimeError(
        "No segmented object found."
    )

largest_contour = max(
    contours,
    key=cv2.contourArea
)


# ============================================================
# 6. CREATE CLEAN MASK
# ============================================================

clean_mask = np.zeros_like(binary_mask)

cv2.drawContours(
    clean_mask,
    [largest_contour],
    -1,
    255,
    thickness=cv2.FILLED
)

cv2.imwrite(
    OUTPUT_CLEAN_MASK,
    clean_mask
)


# ============================================================
# 7. ISOLATE LENS
# ============================================================

isolated_lens = cv2.bitwise_and(
    image,
    image,
    mask=clean_mask
)

cv2.imwrite(
    OUTPUT_ISOLATED,
    isolated_lens
)


# ============================================================
# 8. PIXEL MEASUREMENTS
# ============================================================

area_pixels = cv2.contourArea(
    largest_contour
)

x, y, width_pixels, height_pixels = cv2.boundingRect(
    largest_contour
)


# ============================================================
# 9. CONVERT CONTOUR TO MILLIMETERS
# ============================================================

contour_points = largest_contour.reshape(
    -1,
    2
).astype(np.float32)

physical_points = contour_points.copy()

physical_points[:, 0] = (
    physical_points[:, 0]
    /
    pixels_per_mm_x
)

physical_points[:, 1] = (
    physical_points[:, 1]
    /
    pixels_per_mm_y
)

physical_contour = physical_points.reshape(
    -1,
    1,
    2
)


# ============================================================
# 10. MEASURE REAL-WORLD WIDTH AND HEIGHT
# ============================================================

rect = cv2.minAreaRect(
    physical_contour
)

dimension_1_mm = rect[1][0]
dimension_2_mm = rect[1][1]

lens_width_mm = max(
    dimension_1_mm,
    dimension_2_mm
)

lens_height_mm = min(
    dimension_1_mm,
    dimension_2_mm
)


# ============================================================
# 11. AREA IN MM^2
# ============================================================

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
# 12. DRAW MEASUREMENT BOX
# ============================================================

pixel_rect = cv2.minAreaRect(
    largest_contour
)

box = cv2.boxPoints(
    pixel_rect
)

box = box.astype(int)

measured_image = image.copy()

cv2.drawContours(
    measured_image,
    [box],
    0,
    (0, 255, 0),
    2
)

cv2.putText(
    measured_image,
    f"Width: {lens_width_mm:.2f} mm",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    measured_image,
    f"Height: {lens_height_mm:.2f} mm",
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.imwrite(
    OUTPUT_MEASURED,
    measured_image
)


# ============================================================
# 13. RESULTS
# ============================================================

print()
print("================================")
print("       LENS MEASUREMENT")
print("================================")
print()

print(
    "Pixel width:",
    width_pixels,
    "px"
)

print(
    "Pixel height:",
    height_pixels,
    "px"
)

print()

print(
    "Lens width:",
    round(lens_width_mm, 2),
    "mm"
)

print(
    "Lens height:",
    round(lens_height_mm, 2),
    "mm"
)

print(
    "Lens area:",
    round(area_mm2, 2),
    "mm^2"
)

print()
print("SUCCESS")

print(
    "Measured image saved to:",
    OUTPUT_MEASURED
)
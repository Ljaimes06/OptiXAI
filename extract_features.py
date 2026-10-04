import cv2
import numpy as np


# --------------------------------------------------
# 1. Load rectified image and clean mask
# --------------------------------------------------

image_path = "output/rectified.webp"
mask_path = "output/clean_mask.webp"

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Could not load rectified image.")

if mask is None:
    raise FileNotFoundError("Could not load clean mask.")


# --------------------------------------------------
# 2. Make sure mask is binary
# --------------------------------------------------

_, binary_mask = cv2.threshold(
    mask,
    127,
    255,
    cv2.THRESH_BINARY
)


# --------------------------------------------------
# 3. Find the lens contour
# --------------------------------------------------

contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise RuntimeError("No lens contour found.")

largest_contour = max(contours, key=cv2.contourArea)


# --------------------------------------------------
# 4. Basic contour measurements
# --------------------------------------------------

area_pixels = cv2.contourArea(largest_contour)

perimeter = cv2.arcLength(
    largest_contour,
    True
)

if perimeter > 0:
    circularity = (
        4 * np.pi * area_pixels
    ) / (perimeter ** 2)
else:
    circularity = 0


# --------------------------------------------------
# 5. Centroid
# --------------------------------------------------

moments = cv2.moments(largest_contour)

if moments["m00"] != 0:
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])
else:
    centroid_x = 0
    centroid_y = 0


# --------------------------------------------------
# 6. ROTATED bounding rectangle
# --------------------------------------------------

rect = cv2.minAreaRect(largest_contour)

(center_x, center_y), (rect_width, rect_height), angle = rect


# Put the larger dimension first as width
lens_width_pixels = max(rect_width, rect_height)
lens_height_pixels = min(rect_width, rect_height)


# Get the 4 corners of the rotated rectangle
box = cv2.boxPoints(rect)
box = box.astype(int)


# --------------------------------------------------
# 7. Brightness measurements
# --------------------------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

lens_pixels = gray[binary_mask == 255]

mean_brightness = np.mean(lens_pixels)
brightness_std = np.std(lens_pixels)


# --------------------------------------------------
# 8. Draw measurement box for visualization
# --------------------------------------------------

visualization = image.copy()

cv2.drawContours(
    visualization,
    [box],
    0,
    (0, 255, 0),
    2
)

cv2.circle(
    visualization,
    (centroid_x, centroid_y),
    4,
    (0, 0, 255),
    -1
)

cv2.imwrite(
    "output/lens_measurement_box.webp",
    visualization
)


# --------------------------------------------------
# 9. Print results
# --------------------------------------------------

print()
print("----- LENS FEATURES -----")
print()

print("Area:", round(area_pixels, 2), "pixels^2")

print(
    "Rotated width:",
    round(lens_width_pixels, 2),
    "pixels"
)

print(
    "Rotated height:",
    round(lens_height_pixels, 2),
    "pixels"
)

print(
    "Centroid:",
    centroid_x,
    centroid_y
)

print(
    "Perimeter:",
    round(perimeter, 2)
)

print(
    "Circularity:",
    round(circularity, 3)
)

print(
    "Mean brightness:",
    round(mean_brightness, 2)
)

print(
    "Brightness variation:",
    round(brightness_std, 2)
)

print(
    "Lens rotation angle:",
    round(angle, 2),
    "degrees"
)

print()
print(
    "Measurement image saved to:",
    "output/lens_measurement_box.webp"
)
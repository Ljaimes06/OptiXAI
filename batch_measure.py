import cv2
import numpy as np
import torch
import csv
import os
from pathlib import Path
from PIL import Image
from transformers import Sam2Processor, Sam2Model


# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = r"C:\Users\luisf\PycharmProjects\optiframe\input\lens_2_left"
OUTPUT_FOLDER = "output/batch_lens_1"

MARKER_SIZE_MM = 37.2

# Actual dimensions of this lens
ACTUAL_WIDTH_MM = 54.0
ACTUAL_HEIGHT_MM = 36.0

MODEL_NAME = "facebook/sam2.1-hiera-tiny"


# Broad plausible eyeglass-lens dimensions.
# These are NOT the known 51 x 34 dimensions.
MIN_LENS_WIDTH_MM = 35
MAX_LENS_WIDTH_MM = 85

MIN_LENS_HEIGHT_MM = 20
MAX_LENS_HEIGHT_MM = 65


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD SAM 2
# ============================================================

print("Loading SAM 2...")

processor = Sam2Processor.from_pretrained(
    MODEL_NAME
)

model = Sam2Model.from_pretrained(
    MODEL_NAME
)

model.eval()

print("SAM 2 loaded.\n")


# ============================================================
# ARUCO DETECTOR
# ============================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)


# ============================================================
# FIND IMAGES
# ============================================================

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

image_files = sorted(
    [
        path
        for path in Path(INPUT_FOLDER).iterdir()
        if path.suffix.lower() in valid_extensions
    ]
)

if not image_files:
    raise RuntimeError(
        "No images found in input folder."
    )

print(
    "Images found:",
    len(image_files)
)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, image_path in enumerate(
    image_files,
    start=1
):

    print()
    print("=" * 60)
    print(
        f"Processing {index}/{len(image_files)}:"
    )
    print(image_path.name)
    print("=" * 60)

    try:

        # ====================================================
        # 1. LOAD IMAGE
        # ====================================================

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                "Could not load image."
            )


        # ====================================================
        # 2. ARUCO DETECTION
        # ====================================================

        # ====================================================
        # 2. ARUCO DETECTION
        # ====================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, rejected = detector.detectMarkers(
            gray
        )

        # ----------------------------------------------------
        # If fewer than 4 markers are detected,
        # retry with a white border around the image.
        #
        # This helps when an ArUco marker is very close
        # to the edge of the photograph.
        # ----------------------------------------------------

        if ids is None or len(ids) < 4:

            print(
                "Initial detection found",
                0 if ids is None else len(ids),
                "markers. Retrying with padding..."
            )

            PADDING = 30

            padded_image = cv2.copyMakeBorder(
                image,
                PADDING,
                PADDING,
                PADDING,
                PADDING,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )

            padded_gray = cv2.cvtColor(
                padded_image,
                cv2.COLOR_BGR2GRAY
            )

            padded_corners, padded_ids, padded_rejected = (
                detector.detectMarkers(
                    padded_gray
                )
            )

            if padded_ids is not None:

                # Convert padded-image coordinates
                # back into original-image coordinates
                adjusted_corners = []

                for marker in padded_corners:
                    adjusted = marker.copy()

                    adjusted[:, :, 0] -= PADDING
                    adjusted[:, :, 1] -= PADDING

                    adjusted_corners.append(
                        adjusted
                    )

                corners = adjusted_corners
                ids = padded_ids
                rejected = padded_rejected

        # ----------------------------------------------------
        # Final check
        # ----------------------------------------------------

        if ids is None:
            raise RuntimeError(
                "No ArUco markers detected."
            )

        if len(ids) < 4:
            raise RuntimeError(
                f"Only {len(ids)} ArUco markers detected."
            )

        print(
            "Markers:",
            ids.flatten()
        )

        # ====================================================
        # 3. FIND MARKER CENTERS
        # ====================================================

        marker_centers = []

        for marker in corners:

            points = marker.reshape(
                4,
                2
            )

            center = np.mean(
                points,
                axis=0
            )

            marker_centers.append(
                center
            )


        marker_centers = np.array(
            marker_centers,
            dtype=np.float32
        )


        # ====================================================
        # 4. ORDER FOUR MARKERS
        # ====================================================

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


        top_left = ordered[0]
        top_right = ordered[1]
        bottom_right = ordered[2]
        bottom_left = ordered[3]


        # ====================================================
        # 5. RECTIFIED SIZE
        # ====================================================

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


        # ====================================================
        # 6. PERSPECTIVE TRANSFORM
        # ====================================================

        destination = np.array(
            [
                [0, 0],
                [
                    output_width - 1,
                    0
                ],
                [
                    output_width - 1,
                    output_height - 1
                ],
                [
                    0,
                    output_height - 1
                ]
            ],
            dtype=np.float32
        )


        matrix = cv2.getPerspectiveTransform(
            ordered,
            destination
        )


        rectified = cv2.warpPerspective(
            image,
            matrix,
            (
                output_width,
                output_height
            )
        )


        # ====================================================
        # 7. CALIBRATION
        # ====================================================

        horizontal_sides = []
        vertical_sides = []


        for marker in corners:

            points = marker.reshape(
                4,
                2
            ).astype(np.float32)

            transformed = cv2.perspectiveTransform(
                points.reshape(
                    1,
                    -1,
                    2
                ),
                matrix
            )[0]


            top_side = np.linalg.norm(
                transformed[1]
                -
                transformed[0]
            )

            bottom_side = np.linalg.norm(
                transformed[2]
                -
                transformed[3]
            )

            left_side = np.linalg.norm(
                transformed[3]
                -
                transformed[0]
            )

            right_side = np.linalg.norm(
                transformed[2]
                -
                transformed[1]
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


        pixels_per_mm_x = (
            np.mean(horizontal_sides)
            /
            MARKER_SIZE_MM
        )

        pixels_per_mm_y = (
            np.mean(vertical_sides)
            /
            MARKER_SIZE_MM
        )


        # ====================================================
        # 8. SAVE RECTIFIED IMAGE
        # ====================================================

        stem = image_path.stem

        rectified_path = os.path.join(
            OUTPUT_FOLDER,
            f"{stem}_rectified.jpg"
        )

        cv2.imwrite(
            rectified_path,
            rectified
        )


        # ====================================================
        # 9. PREPARE IMAGE FOR SAM
        # ====================================================

        rgb = cv2.cvtColor(
            rectified,
            cv2.COLOR_BGR2RGB
        )

        pil_image = Image.fromarray(
            rgb
        )

        width, height = pil_image.size


        # ====================================================
        # 10. CREATE SEARCH POINTS THROUGHOUT IMAGE
        #
        # Instead of assuming lens is centered,
        # search across the usable image area.
        # ====================================================

        x_positions = np.linspace(
            width * 0.15,
            width * 0.85,
            5
        )

        y_positions = np.linspace(
            height * 0.12,
            height * 0.88,
            7
        )


        object_points = []
        object_labels = []


        for y_point in y_positions:

            for x_point in x_positions:

                object_points.append(
                    [
                        [
                            int(x_point),
                            int(y_point)
                        ]
                    ]
                )

                object_labels.append(
                    [1]
                )


        input_points = [
            object_points
        ]

        input_labels = [
            object_labels
        ]


        # ====================================================
        # 11. RUN SAM FOR ALL SEARCH POINTS
        # ====================================================

        inputs = processor(
            images=pil_image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt"
        )


        with torch.no_grad():

            outputs = model(
                **inputs,
                multimask_output=False
            )


        processed_masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )


        masks = processed_masks[0]


        # ====================================================
        # 12. CHOOSE BEST LENS MASK
        # ====================================================

        best_mask = None
        best_contour = None
        best_score = -999999

        candidates_found = 0


        for object_index in range(
            masks.shape[0]
        ):

            candidate_mask = masks[
                object_index
            ][0]

            candidate_mask = (
                candidate_mask.cpu().numpy()
                > 0
            ).astype(
                np.uint8
            ) * 255


            candidate_contours, _ = cv2.findContours(
                candidate_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not candidate_contours:
                continue


            contour = max(
                candidate_contours,
                key=cv2.contourArea
            )


            if cv2.contourArea(contour) < 100:
                continue


            # --------------------------------------------
            # Reject regions touching image border
            # --------------------------------------------

            x, y, w, h = cv2.boundingRect(
                contour
            )

            margin = 3

            if (
                x <= margin
                or
                y <= margin
                or
                x + w >= width - margin
                or
                y + h >= height - margin
            ):
                continue


            # --------------------------------------------
            # Convert candidate into millimeters
            # --------------------------------------------

            contour_points = contour.reshape(
                -1,
                2
            ).astype(
                np.float32
            )


            physical_points = (
                contour_points.copy()
            )


            physical_points[:, 0] /= (
                pixels_per_mm_x
            )

            physical_points[:, 1] /= (
                pixels_per_mm_y
            )


            physical_contour = (
                physical_points.reshape(
                    -1,
                    1,
                    2
                )
            )


            physical_rect = cv2.minAreaRect(
                physical_contour
            )


            dim1 = physical_rect[1][0]
            dim2 = physical_rect[1][1]


            candidate_width = max(
                dim1,
                dim2
            )

            candidate_height = min(
                dim1,
                dim2
            )


            # --------------------------------------------
            # Reject impossible lens sizes
            # --------------------------------------------

            if not (
                MIN_LENS_WIDTH_MM
                <= candidate_width
                <= MAX_LENS_WIDTH_MM
            ):
                continue


            if not (
                MIN_LENS_HEIGHT_MM
                <= candidate_height
                <= MAX_LENS_HEIGHT_MM
            ):
                continue


            # --------------------------------------------
            # Shape check
            #
            # Lens should occupy a reasonable fraction
            # of its surrounding rectangle.
            # --------------------------------------------

            rect_area = (
                physical_rect[1][0]
                *
                physical_rect[1][1]
            )

            if rect_area <= 0:
                continue


            contour_area_mm = (
                cv2.contourArea(contour)
                /
                (
                    pixels_per_mm_x
                    *
                    pixels_per_mm_y
                )
            )


            fill_ratio = (
                contour_area_mm
                /
                rect_area
            )


            if not (
                0.40
                <= fill_ratio
                <= 0.95
            ):
                continue


            candidates_found += 1


            # --------------------------------------------
            # SAM confidence
            # --------------------------------------------

            confidence = 0.0

            if hasattr(
                outputs,
                "iou_scores"
            ):

                try:

                    confidence = float(
                        outputs.iou_scores[
                            0,
                            object_index,
                            0
                        ]
                    )

                except Exception:
                    confidence = 0.0


            # --------------------------------------------
            # Score candidate
            #
            # Elliptical / lens-like region tends to have
            # fill ratio around ~0.7-0.8.
            # --------------------------------------------

            shape_score = (
                1.0
                -
                abs(
                    fill_ratio
                    -
                    0.75
                )
            )


            score = (
                confidence * 2.0
                +
                shape_score
            )


            if score > best_score:

                best_score = score

                best_mask = (
                    candidate_mask.copy()
                )

                best_contour = (
                    contour.copy()
                )


        print(
            "Plausible SAM masks:",
            candidates_found
        )


        if best_mask is None:
            raise RuntimeError(
                "Could not identify a plausible lens mask."
            )


        # ====================================================
        # 13. CLEAN FINAL MASK
        # ====================================================

        clean_mask = np.zeros_like(
            best_mask
        )

        cv2.drawContours(
            clean_mask,
            [best_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )


        mask_path = os.path.join(
            OUTPUT_FOLDER,
            f"{stem}_mask.png"
        )

        cv2.imwrite(
            mask_path,
            clean_mask
        )


        # ====================================================
        # 14. CONVERT FINAL CONTOUR TO MILLIMETERS
        # ====================================================

        contour_points = best_contour.reshape(
            -1,
            2
        ).astype(
            np.float32
        )


        physical_points = (
            contour_points.copy()
        )


        physical_points[:, 0] /= (
            pixels_per_mm_x
        )

        physical_points[:, 1] /= (
            pixels_per_mm_y
        )


        physical_contour = (
            physical_points.reshape(
                -1,
                1,
                2
            )
        )


        physical_rect = cv2.minAreaRect(
            physical_contour
        )


        dim1 = physical_rect[1][0]
        dim2 = physical_rect[1][1]


        measured_width = max(
            dim1,
            dim2
        )

        measured_height = min(
            dim1,
            dim2
        )


        # ====================================================
        # 15. CALCULATE ERROR
        # ====================================================

        width_error = abs(
            measured_width
            -
            ACTUAL_WIDTH_MM
        )

        height_error = abs(
            measured_height
            -
            ACTUAL_HEIGHT_MM
        )


        width_percent_error = (
            width_error
            /
            ACTUAL_WIDTH_MM
        ) * 100


        height_percent_error = (
            height_error
            /
            ACTUAL_HEIGHT_MM
        ) * 100


        # ====================================================
        # 16. DRAW FINAL RESULT
        # ====================================================

        pixel_rect = cv2.minAreaRect(
            best_contour
        )

        box = cv2.boxPoints(
            pixel_rect
        ).astype(
            int
        )


        measured_image = (
            rectified.copy()
        )


        cv2.drawContours(
            measured_image,
            [box],
            0,
            (0, 255, 0),
            2
        )


        cv2.putText(
            measured_image,
            f"W: {measured_width:.2f} mm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            measured_image,
            f"H: {measured_height:.2f} mm",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        measured_path = os.path.join(
            OUTPUT_FOLDER,
            f"{stem}_measured.jpg"
        )


        cv2.imwrite(
            measured_path,
            measured_image
        )


        # ====================================================
        # 17. SAVE RESULT
        # ====================================================

        results.append(
            {
                "image":
                    image_path.name,

                "width_mm":
                    measured_width,

                "height_mm":
                    measured_height,

                "width_error_mm":
                    width_error,

                "height_error_mm":
                    height_error,

                "width_error_percent":
                    width_percent_error,

                "height_error_percent":
                    height_percent_error,

                "pixels_per_mm_x":
                    pixels_per_mm_x,

                "pixels_per_mm_y":
                    pixels_per_mm_y,

                "status":
                    "SUCCESS"
            }
        )


        print(
            f"Width:  {measured_width:.2f} mm"
        )

        print(
            f"Height: {measured_height:.2f} mm"
        )

        print(
            f"Width error:  {width_error:.2f} mm"
        )

        print(
            f"Height error: {height_error:.2f} mm"
        )


    except Exception as error:

        print(
            "FAILED:",
            error
        )

        results.append(
            {
                "image":
                    image_path.name,

                "width_mm":
                    "",

                "height_mm":
                    "",

                "width_error_mm":
                    "",

                "height_error_mm":
                    "",

                "width_error_percent":
                    "",

                "height_error_percent":
                    "",

                "pixels_per_mm_x":
                    "",

                "pixels_per_mm_y":
                    "",

                "status":
                    str(error)
            }
        )


# ============================================================
# 18. SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_FOLDER,
    "results.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as file:

    fieldnames = [
        "image",
        "width_mm",
        "height_mm",
        "width_error_mm",
        "height_error_mm",
        "width_error_percent",
        "height_error_percent",
        "pixels_per_mm_x",
        "pixels_per_mm_y",
        "status"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

successful = [
    result
    for result in results
    if result["status"] == "SUCCESS"
]


if successful:

    widths = np.array(
        [
            result["width_mm"]
            for result in successful
        ],
        dtype=float
    )

    heights = np.array(
        [
            result["height_mm"]
            for result in successful
        ],
        dtype=float
    )


    width_errors = np.abs(
        widths
        -
        ACTUAL_WIDTH_MM
    )

    height_errors = np.abs(
        heights
        -
        ACTUAL_HEIGHT_MM
    )


    print()
    print("=" * 60)
    print("               FINAL BATCH RESULTS")
    print("=" * 60)

    print(
        "Successful images:",
        len(successful),
        "/",
        len(image_files)
    )

    print()

    print(
        "Actual lens:",
        ACTUAL_WIDTH_MM,
        "x",
        ACTUAL_HEIGHT_MM,
        "mm"
    )

    print()

    print(
        "Average width:",
        round(
            np.mean(widths),
            2
        ),
        "mm"
    )

    print(
        "Average height:",
        round(
            np.mean(heights),
            2
        ),
        "mm"
    )

    print()

    print(
        "Average width error:",
        round(
            np.mean(width_errors),
            2
        ),
        "mm"
    )

    print(
        "Average height error:",
        round(
            np.mean(height_errors),
            2
        ),
        "mm"
    )

    print()

    print(
        "Worst width error:",
        round(
            np.max(width_errors),
            2
        ),
        "mm"
    )

    print(
        "Worst height error:",
        round(
            np.max(height_errors),
            2
        ),
        "mm"
    )


    if len(widths) > 1:

        print()

        print(
            "Width standard deviation:",
            round(
                np.std(
                    widths,
                    ddof=1
                ),
                3
            ),
            "mm"
        )

        print(
            "Height standard deviation:",
            round(
                np.std(
                    heights,
                    ddof=1
                ),
                3
            ),
            "mm"
        )


    print()
    print(
        "CSV saved to:",
        csv_path
    )

else:

    print(
        "No images were processed successfully."
    )
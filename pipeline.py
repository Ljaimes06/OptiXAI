import cv2
import numpy as np
import torch
import os

from PIL import Image
from transformers import Sam2Processor, Sam2Model


# ============================================================
# SETTINGS
# ============================================================

MARKER_SIZE_MM = 37.2

MODEL_NAME = "facebook/sam2.1-hiera-tiny"

# Broad plausible eyeglass-lens dimensions
MIN_LENS_WIDTH_MM = 35
MAX_LENS_WIDTH_MM = 85

MIN_LENS_HEIGHT_MM = 20
MAX_LENS_HEIGHT_MM = 65


# ============================================================
# LOAD SAM 2 ONCE
# ============================================================

print("Loading SAM 2...")

processor = Sam2Processor.from_pretrained(
    MODEL_NAME
)

model = Sam2Model.from_pretrained(
    MODEL_NAME
)

model.eval()

print("SAM 2 loaded.")


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
# SVG EXPORT FUNCTION
# ============================================================

def create_svg_from_contour(
    contour_mm,
    output_path,
    margin_mm=5
):

    # Make a copy so original contour is not modified
    points = np.array(
        contour_mm,
        dtype=np.float32
    ).copy()

    if len(points) < 3:
        raise RuntimeError(
            "Not enough contour points to create SVG."
        )

    # --------------------------------------------------------
    # Move contour close to SVG origin
    # --------------------------------------------------------

    min_x = np.min(
        points[:, 0]
    )

    min_y = np.min(
        points[:, 1]
    )

    points[:, 0] -= min_x
    points[:, 1] -= min_y


    # --------------------------------------------------------
    # Add printing margin
    # --------------------------------------------------------

    points[:, 0] += margin_mm
    points[:, 1] += margin_mm


    # --------------------------------------------------------
    # SVG physical dimensions in MILLIMETERS
    # --------------------------------------------------------

    width_mm = (
        np.max(points[:, 0])
        +
        margin_mm
    )

    height_mm = (
        np.max(points[:, 1])
        +
        margin_mm
    )


    # --------------------------------------------------------
    # Convert contour points to SVG polygon string
    # --------------------------------------------------------

    point_string = " ".join(
        f"{x:.3f},{y:.3f}"
        for x, y in points
    )


    # --------------------------------------------------------
    # Build SVG
    #
    # width and height use mm explicitly,
    # which allows 1:1 physical printing.
    # --------------------------------------------------------

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg
    xmlns="http://www.w3.org/2000/svg"
    width="{width_mm:.3f}mm"
    height="{height_mm:.3f}mm"
    viewBox="0 0 {width_mm:.3f} {height_mm:.3f}">

    <polygon
        points="{point_string}"
        fill="none"
        stroke="black"
        stroke-width="0.3"
        stroke-linejoin="round"
    />

</svg>
"""


    # --------------------------------------------------------
    # Save SVG
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            svg
        )


    return output_path


# ============================================================
# MAIN MEASUREMENT FUNCTION
# ============================================================

def measure_lens(
    image_path,
    output_folder="output/app"
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    # ========================================================
    # 1. LOAD IMAGE
    # ========================================================

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            "Could not load image."
        )


    original_path = os.path.join(
        output_folder,
        "original.jpg"
    )

    cv2.imwrite(
        original_path,
        image
    )


    # ========================================================
    # 2. DETECT ARUCO MARKERS
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, rejected = detector.detectMarkers(
        gray
    )


    # ========================================================
    # 3. FALLBACK FOR MARKERS NEAR IMAGE EDGE
    # ========================================================

    used_padding = False


    if ids is None or len(ids) < 4:

        used_padding = True

        padding = 30


        padded_image = cv2.copyMakeBorder(
            image,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(
                255,
                255,
                255
            )
        )


        padded_gray = cv2.cvtColor(
            padded_image,
            cv2.COLOR_BGR2GRAY
        )


        (
            padded_corners,
            padded_ids,
            padded_rejected
        ) = detector.detectMarkers(
            padded_gray
        )


        if padded_ids is not None:

            adjusted_corners = []


            for marker in padded_corners:

                adjusted = marker.copy()

                adjusted[:, :, 0] -= padding
                adjusted[:, :, 1] -= padding

                adjusted_corners.append(
                    adjusted
                )


            corners = adjusted_corners
            ids = padded_ids
            rejected = padded_rejected


    # ========================================================
    # 4. VERIFY FOUR MARKERS
    # ========================================================

    if ids is None:

        raise RuntimeError(
            "No ArUco markers detected."
        )


    if len(ids) < 4:

        raise RuntimeError(
            f"Only {len(ids)} ArUco markers detected. "
            "Make sure all four markers are visible."
        )


    # ========================================================
    # 5. DRAW DETECTED MARKERS
    # ========================================================

    aruco_image = image.copy()


    cv2.aruco.drawDetectedMarkers(
        aruco_image,
        corners,
        ids
    )


    aruco_path = os.path.join(
        output_folder,
        "aruco_detected.jpg"
    )


    cv2.imwrite(
        aruco_path,
        aruco_image
    )


    # ========================================================
    # 6. FIND MARKER CENTERS
    # ========================================================

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


    # ========================================================
    # 7. ORDER MARKERS
    #
    # TL
    # TR
    # BR
    # BL
    # ========================================================

    ordered = np.zeros(
        (
            4,
            2
        ),
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
        np.argmin(
            sums
        )
    ]

    ordered[2] = marker_centers[
        np.argmax(
            sums
        )
    ]

    ordered[1] = marker_centers[
        np.argmin(
            differences
        )
    ]

    ordered[3] = marker_centers[
        np.argmax(
            differences
        )
    ]


    top_left = ordered[0]
    top_right = ordered[1]
    bottom_right = ordered[2]
    bottom_left = ordered[3]


    # ========================================================
    # 8. RECTIFIED IMAGE SIZE
    # ========================================================

    top_width = np.linalg.norm(
        top_right
        -
        top_left
    )


    bottom_width = np.linalg.norm(
        bottom_right
        -
        bottom_left
    )


    left_height = np.linalg.norm(
        bottom_left
        -
        top_left
    )


    right_height = np.linalg.norm(
        bottom_right
        -
        top_right
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


    if (
        output_width <= 0
        or
        output_height <= 0
    ):

        raise RuntimeError(
            "Invalid rectified image dimensions."
        )


    # ========================================================
    # 9. DESTINATION RECTANGLE
    # ========================================================

    destination = np.array(
        [
            [
                0,
                0
            ],

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


    # ========================================================
    # 10. PERSPECTIVE TRANSFORM
    # ========================================================

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


    rectified_path = os.path.join(
        output_folder,
        "rectified.jpg"
    )


    cv2.imwrite(
        rectified_path,
        rectified
    )


    # ========================================================
    # 11. PIXEL / MILLIMETER CALIBRATION
    # ========================================================

    horizontal_sides = []
    vertical_sides = []


    for marker in corners:

        points = marker.reshape(
            4,
            2
        ).astype(
            np.float32
        )


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
        np.mean(
            horizontal_sides
        )
        /
        MARKER_SIZE_MM
    )


    pixels_per_mm_y = (
        np.mean(
            vertical_sides
        )
        /
        MARKER_SIZE_MM
    )


    # ========================================================
    # 12. PREPARE IMAGE FOR SAM 2
    # ========================================================

    rgb = cv2.cvtColor(
        rectified,
        cv2.COLOR_BGR2RGB
    )


    pil_image = Image.fromarray(
        rgb
    )


    width, height = pil_image.size


    # ========================================================
    # 13. SEARCH POINT GRID
    # ========================================================

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
                        int(
                            x_point
                        ),
                        int(
                            y_point
                        )
                    ]
                ]
            )


            object_labels.append(
                [
                    1
                ]
            )


    input_points = [
        object_points
    ]


    input_labels = [
        object_labels
    ]


    # ========================================================
    # 14. RUN SAM 2
    # ========================================================

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


    processed_masks = (
        processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs[
                "original_sizes"
            ]
        )
    )


    masks = processed_masks[0]


    # ========================================================
    # 15. CHOOSE BEST LENS MASK
    # ========================================================

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
            candidate_mask
            .cpu()
            .numpy()
            > 0
        ).astype(
            np.uint8
        ) * 255


        (
            candidate_contours,
            _
        ) = cv2.findContours(
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


        if cv2.contourArea(
            contour
        ) < 100:

            continue


        # ----------------------------------------------------
        # Reject contours touching edge
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Convert candidate to millimeters
        # ----------------------------------------------------

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


        dimension_1 = (
            physical_rect[1][0]
        )

        dimension_2 = (
            physical_rect[1][1]
        )


        candidate_width = max(
            dimension_1,
            dimension_2
        )


        candidate_height = min(
            dimension_1,
            dimension_2
        )


        # ----------------------------------------------------
        # Physical size filter
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Shape filter
        # ----------------------------------------------------

        rect_area = (
            physical_rect[1][0]
            *
            physical_rect[1][1]
        )


        if rect_area <= 0:

            continue


        contour_area_mm2 = (
            cv2.contourArea(
                contour
            )
            /
            (
                pixels_per_mm_x
                *
                pixels_per_mm_y
            )
        )


        fill_ratio = (
            contour_area_mm2
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


        # ----------------------------------------------------
        # SAM confidence
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Shape preference
        # ----------------------------------------------------

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


    if best_mask is None:

        raise RuntimeError(
            "Could not identify a plausible lens."
        )


    # ========================================================
    # 16. CLEAN FINAL MASK
    # ========================================================

    clean_mask = np.zeros_like(
        best_mask
    )


    cv2.drawContours(
        clean_mask,
        [
            best_contour
        ],
        -1,
        255,
        thickness=cv2.FILLED
    )


    mask_path = os.path.join(
        output_folder,
        "lens_mask.png"
    )


    cv2.imwrite(
        mask_path,
        clean_mask
    )


    # ========================================================
    # 17. FINAL CONTOUR IN MILLIMETERS
    # ========================================================

    contour_points = (
        best_contour
        .reshape(
            -1,
            2
        )
        .astype(
            np.float32
        )
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


    # ========================================================
    # 18. A / B BOXING DIMENSIONS
    # ========================================================

    physical_rect = cv2.minAreaRect(
        physical_contour
    )


    dimension_1_mm = (
        physical_rect[1][0]
    )


    dimension_2_mm = (
        physical_rect[1][1]
    )


    A_mm = float(
        max(
            dimension_1_mm,
            dimension_2_mm
        )
    )


    B_mm = float(
        min(
            dimension_1_mm,
            dimension_2_mm
        )
    )


    # ========================================================
    # 19. PERIMETER
    # ========================================================

    perimeter_mm = float(
        cv2.arcLength(
            physical_contour,
            True
        )
    )


    # ========================================================
    # 20. AREA
    # ========================================================

    area_mm2 = float(
        cv2.contourArea(
            physical_contour
        )
    )


    # ========================================================
    # 21. EXPORT 1:1 SVG
    # ========================================================

    svg_path = os.path.join(
        output_folder,
        "lens_contour.svg"
    )


    create_svg_from_contour(
        physical_points,
        svg_path
    )


    # ========================================================
    # 22. DRAW FINAL MEASUREMENT
    # ========================================================

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


    # Green boxing rectangle
    cv2.drawContours(
        measured_image,
        [
            box
        ],
        0,
        (
            0,
            255,
            0
        ),
        2
    )


    # Blue actual lens contour
    cv2.drawContours(
        measured_image,
        [
            best_contour
        ],
        -1,
        (
            255,
            0,
            0
        ),
        2
    )


    # --------------------------------------------------------
    # Measurement text
    # --------------------------------------------------------

    cv2.putText(
        measured_image,
        f"A: {A_mm:.2f} mm",
        (
            10,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            0,
            255,
            0
        ),
        2
    )


    cv2.putText(
        measured_image,
        f"B: {B_mm:.2f} mm",
        (
            10,
            60
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            0,
            255,
            0
        ),
        2
    )


    cv2.putText(
        measured_image,
        f"P: {perimeter_mm:.2f} mm",
        (
            10,
            90
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            0,
            255,
            0
        ),
        2
    )


    measured_path = os.path.join(
        output_folder,
        "measured.jpg"
    )


    cv2.imwrite(
        measured_path,
        measured_image
    )


    # ========================================================
    # 23. RETURN RESULTS
    # ========================================================

    return {

        # ----------------------------------------------------
        # Required lens measurements
        # ----------------------------------------------------

        "A_mm":
            round(
                A_mm,
                2
            ),

        "B_mm":
            round(
                B_mm,
                2
            ),

        "perimeter_mm":
            round(
                perimeter_mm,
                2
            ),

        "area_mm2":
            round(
                area_mm2,
                2
            ),


        # ----------------------------------------------------
        # Compatibility aliases
        # ----------------------------------------------------

        "width_mm":
            round(
                A_mm,
                2
            ),

        "height_mm":
            round(
                B_mm,
                2
            ),


        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        "pixels_per_mm_x":
            round(
                float(
                    pixels_per_mm_x
                ),
                4
            ),

        "pixels_per_mm_y":
            round(
                float(
                    pixels_per_mm_y
                ),
                4
            ),


        # ----------------------------------------------------
        # Marker information
        # ----------------------------------------------------

        "markers_detected":
            int(
                len(
                    ids
                )
            ),

        "marker_ids":
            [
                int(
                    marker_id
                )
                for marker_id
                in ids.flatten()
            ],

        "padding_used":
            used_padding,


        # ----------------------------------------------------
        # AI information
        # ----------------------------------------------------

        "sam_candidates":
            int(
                candidates_found
            ),


        # ----------------------------------------------------
        # Physical contour
        #
        # Used later for frame/STL generation.
        # ----------------------------------------------------

        "contour_mm":
            physical_points.tolist(),


        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        "original_path":
            original_path,

        "aruco_path":
            aruco_path,

        "rectified_path":
            rectified_path,

        "mask_path":
            mask_path,

        "measured_path":
            measured_path,

        "svg_path":
            svg_path
    }
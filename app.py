import os
import tempfile

import streamlit as st

from pipeline import measure_lens


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="OptiXAI",
    page_icon="👓",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("👓 OptiXAI")

st.write(
    "Measure recycled eyeglass lenses using AI and computer vision."
)

st.write(
    "Place the lens inside the ArUco reference frame, "
    "then take a photo or upload an existing image."
)


# ============================================================
# LENS SIDE
# ============================================================

eye_side = st.radio(
    "Which lens are you measuring?",
    [
        "Left lens",
        "Right lens"
    ],
    horizontal=True
)


# ============================================================
# CAMERA
# ============================================================

st.subheader("Take a picture")

camera_photo = st.camera_input(
    "Position the lens and all four ArUco markers "
    "inside the camera view"
)


# ============================================================
# FILE UPLOAD FALLBACK
# ============================================================

st.write("Or upload an existing image:")

uploaded_file = st.file_uploader(
    "Upload lens image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# CHOOSE IMAGE SOURCE
# ============================================================

if camera_photo is not None:
    image_file = camera_photo
else:
    image_file = uploaded_file


# ============================================================
# PROCESS IMAGE
# ============================================================

if image_file is not None:

    st.subheader("Selected image")

    st.image(
        image_file,
        use_container_width=True
    )


    if st.button(
        "Analyze Lens",
        type="primary",
        use_container_width=True
    ):

        temp_path = None

        try:

            with st.spinner(
                "Detecting markers, correcting perspective, "
                "segmenting the lens and calculating dimensions..."
            ):

                filename = getattr(
                    image_file,
                    "name",
                    "lens.jpg"
                )

                suffix = os.path.splitext(
                    filename
                )[1]

                if not suffix:
                    suffix = ".jpg"


                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        image_file.getbuffer()
                    )

                    temp_path = temp_file.name


                result = measure_lens(
                    temp_path,
                    output_folder="output/app"
                )


            # ====================================================
            # SUCCESS
            # ====================================================

            st.success(
                f"{eye_side} successfully detected and measured."
            )


            # ====================================================
            # MAIN MEASUREMENTS
            # ====================================================

            st.subheader("Lens Measurements")

            col1, col2, col3 = st.columns(3)


            with col1:
                st.metric(
                    "A — Width",
                    f'{result["A_mm"]:.2f} mm'
                )


            with col2:
                st.metric(
                    "B — Height",
                    f'{result["B_mm"]:.2f} mm'
                )


            with col3:
                st.metric(
                    "Perimeter",
                    f'{result["perimeter_mm"]:.2f} mm'
                )


            # ====================================================
            # SVG EXPORT
            # ====================================================

            st.subheader("Export Lens Contour")

            svg_filename = (
                "left_lens_contour.svg"
                if eye_side == "Left lens"
                else "right_lens_contour.svg"
            )

            with open(
                result["svg_path"],
                "rb"
            ) as svg_file:

                st.download_button(
                    label="Download 1:1 SVG Contour",
                    data=svg_file.read(),
                    file_name=svg_filename,
                    mime="image/svg+xml",
                    use_container_width=True
                )

            st.caption(
                "The SVG is exported at real-world 1:1 scale "
                "in millimeters. Print at 100% / actual size."
            )


            # ====================================================
            # FINAL MEASURED IMAGE
            # ====================================================

            st.subheader("Measured Lens")

            st.image(
                result["measured_path"],
                caption=(
                    f"{eye_side}: "
                    f'A = {result["A_mm"]:.2f} mm, '
                    f'B = {result["B_mm"]:.2f} mm, '
                    f'P = {result["perimeter_mm"]:.2f} mm'
                ),
                use_container_width=True
            )


            # ====================================================
            # PIPELINE VISUALIZATION
            # ====================================================

            st.subheader(
                "How OptiXAI processed the image"
            )

            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "1. Reference Detection",
                    "2. Perspective Correction",
                    "3. AI Segmentation",
                    "4. Final Measurement"
                ]
            )


            with tab1:
                st.image(
                    result["aruco_path"],
                    caption=(
                        "Detected ArUco reference markers "
                        "for scale and perspective calibration"
                    ),
                    use_container_width=True
                )


            with tab2:
                st.image(
                    result["rectified_path"],
                    caption="Perspective-corrected top-down image",
                    use_container_width=True
                )


            with tab3:
                st.image(
                    result["mask_path"],
                    caption="Lens isolated using SAM 2 segmentation",
                    use_container_width=True
                )


            with tab4:
                st.image(
                    result["measured_path"],
                    caption="Final lens measurement",
                    use_container_width=True
                )


            # ====================================================
            # TECHNICAL DETAILS
            # ====================================================

            with st.expander("Technical details"):

                st.write(
                    "**Lens side:**",
                    eye_side
                )

                st.write(
                    "**A — Width:**",
                    result["A_mm"],
                    "mm"
                )

                st.write(
                    "**B — Height:**",
                    result["B_mm"],
                    "mm"
                )

                st.write(
                    "**Perimeter:**",
                    result["perimeter_mm"],
                    "mm"
                )

                st.write(
                    "**Area:**",
                    result["area_mm2"],
                    "mm²"
                )

                st.write(
                    "**ArUco markers detected:**",
                    result["markers_detected"]
                )

                st.write(
                    "**Marker IDs:**",
                    result["marker_ids"]
                )

                st.write(
                    "**Horizontal calibration:**",
                    result["pixels_per_mm_x"],
                    "pixels/mm"
                )

                st.write(
                    "**Vertical calibration:**",
                    result["pixels_per_mm_y"],
                    "pixels/mm"
                )

                st.write(
                    "**Plausible SAM masks:**",
                    result["sam_candidates"]
                )

                st.write(
                    "**Edge-padding fallback used:**",
                    result["padding_used"]
                )

                st.write(
                    "**Contour points preserved:**",
                    len(
                        result["contour_mm"]
                    )
                )


        # ========================================================
        # ERROR HANDLING
        # ========================================================

        except Exception as error:

            st.error(
                "OptiXAI could not analyze this image."
            )

            st.warning(
                "Make sure all four ArUco markers are visible, "
                "the image is reasonably sharp, and the lens "
                "is inside the reference area."
            )

            st.write(
                "Technical error:",
                str(error)
            )


        # ========================================================
        # DELETE TEMP FILE
        # ========================================================

        finally:

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):

                os.remove(
                    temp_path
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "OptiXAI — AI-powered measurement of recycled eyeglass lenses."
)

st.caption(
    "Hackathon prototype uses calibrated ArUco markers "
    "as the physical reference."
)
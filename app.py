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

st.title(
    "👓 OptiXAI"
)

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

st.subheader(
    "Take a picture"
)

camera_photo = st.camera_input(
    "Position the lens and all four ArUco markers "
    "inside the camera view"
)


# ============================================================
# FILE UPLOAD FALLBACK
# ============================================================

st.write(
    "Or upload an existing image:"
)

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
#
# Camera has priority.
# ============================================================

if camera_photo is not None:

    image_file = camera_photo

else:

    image_file = uploaded_file


# ============================================================
# PROCESS SELECTED IMAGE
# ============================================================

if image_file is not None:

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    st.subheader(
        "Selected image"
    )

    st.image(
        image_file,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Analyze button
    # --------------------------------------------------------

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

                # ------------------------------------------------
                # Get extension
                # ------------------------------------------------

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


                # ------------------------------------------------
                # Save temporary copy
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        image_file.getbuffer()
                    )

                    temp_path = (
                        temp_file.name
                    )


                # ------------------------------------------------
                # Run OptiXAI pipeline
                # ------------------------------------------------

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

            st.subheader(
                "Lens Measurements"
            )


            col1, col2, col3 = st.columns(
                3
            )


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
            # SECONDARY INFORMATION
            # ====================================================

            with st.expander(
                "Additional measurement information"
            ):

                st.write(
                    "**Lens area:**",
                    f'{result["area_mm2"]:.2f} mm²'
                )


            # ====================================================
            # FINAL MEASURED IMAGE
            # ====================================================

            st.subheader(
                "Measured Lens"
            )


            st.image(
                result["measured_path"],
                caption=(
                    f"{eye_side}: "
                    f'A = {result["A_mm"]:.2f} mm, '
                    f'B = {result["B_mm"]:.2f} mm'
                ),
                use_container_width=True
            )


            # ====================================================
            # PIPELINE VISUALIZATION
            # ====================================================

            st.subheader(
                "How OptiXAI processed the image"
            )


            (
                tab1,
                tab2,
                tab3,
                tab4
            ) = st.tabs(
                [
                    "1. Reference Detection",
                    "2. Perspective Correction",
                    "3. AI Segmentation",
                    "4. Final Measurement"
                ]
            )


            # ----------------------------------------------------
            # ArUco detection
            # ----------------------------------------------------

            with tab1:

                st.image(
                    result["aruco_path"],
                    caption=(
                        "Detected ArUco reference markers "
                        "used for scale and perspective calibration"
                    ),
                    use_container_width=True
                )


            # ----------------------------------------------------
            # Rectification
            # ----------------------------------------------------

            with tab2:

                st.image(
                    result["rectified_path"],
                    caption=(
                        "Perspective-corrected top-down image"
                    ),
                    use_container_width=True
                )


            # ----------------------------------------------------
            # Segmentation
            # ----------------------------------------------------

            with tab3:

                st.image(
                    result["mask_path"],
                    caption=(
                        "Lens isolated using SAM 2 segmentation"
                    ),
                    use_container_width=True
                )


            # ----------------------------------------------------
            # Final measurement
            # ----------------------------------------------------

            with tab4:

                st.image(
                    result["measured_path"],
                    caption=(
                        "Measured lens contour in real-world units"
                    ),
                    use_container_width=True
                )


            # ====================================================
            # TECHNICAL DETAILS
            # ====================================================

            with st.expander(
                "Technical details"
            ):

                st.write(
                    "**Lens side:**",
                    eye_side
                )


                st.write(
                    "**ArUco markers detected:**",
                    result[
                        "markers_detected"
                    ]
                )


                st.write(
                    "**Marker IDs:**",
                    result[
                        "marker_ids"
                    ]
                )


                st.write(
                    "**Horizontal calibration:**",
                    result[
                        "pixels_per_mm_x"
                    ],
                    "pixels/mm"
                )


                st.write(
                    "**Vertical calibration:**",
                    result[
                        "pixels_per_mm_y"
                    ],
                    "pixels/mm"
                )


                st.write(
                    "**Plausible SAM masks:**",
                    result[
                        "sam_candidates"
                    ]
                )


                st.write(
                    "**Marker edge-padding fallback used:**",
                    result[
                        "padding_used"
                    ]
                )


                st.write(
                    "**Contour points preserved:**",
                    len(
                        result[
                            "contour_mm"
                        ]
                    )
                )


        # ========================================================
        # ERRORS
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
                str(
                    error
                )
            )


        # ========================================================
        # REMOVE TEMP FILE
        # ========================================================

        finally:

            if (
                temp_path is not None
                and os.path.exists(
                    temp_path
                )
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
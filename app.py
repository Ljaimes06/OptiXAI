import os
import tempfile
import streamlit as st

from pipeline import measure_lens


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="OptiFrame",
    page_icon="👓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👓 OptiFrame")

st.write(
    "Upload a lens image with the four ArUco reference markers visible."
)

st.write(
    "OptiFrame will automatically rectify the image, segment the lens, "
    "and estimate its physical dimensions."
)


# ============================================================
# FILE UPLOADER
# ============================================================

camera_photo = st.camera_input(
    "Take a picture of the lens"
)

uploaded_file = st.file_uploader(
    "Or upload an existing image",
    type=["jpg", "jpeg", "png", "webp"]
)
# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # ----------------------------------------
    # Show uploaded image
    # ----------------------------------------

    st.subheader("Uploaded image")

    st.image(
        uploaded_file,
        use_container_width=True
    )


    # ----------------------------------------
    # Analyze button
    # ----------------------------------------

    if st.button(
        "Analyze Lens",
        type="primary"
    ):

        try:

            with st.spinner(
                "Analyzing lens..."
            ):

                # ----------------------------------------
                # Save uploaded image temporarily
                # ----------------------------------------

                suffix = os.path.splitext(
                    uploaded_file.name
                )[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = (
                        temp_file.name
                    )


                # ----------------------------------------
                # Run OptiFrame pipeline
                # ----------------------------------------

                result = measure_lens(
                    temp_path,
                    output_folder="output/app"
                )


            # ====================================================
            # RESULTS
            # ====================================================

            st.success(
                "Lens successfully detected and measured."
            )


            # ----------------------------------------
            # Main measurements
            # ----------------------------------------

            st.subheader(
                "Lens Measurements"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Width",
                    f'{result["width_mm"]:.2f} mm'
                )

            with col2:

                st.metric(
                    "Height",
                    f'{result["height_mm"]:.2f} mm'
                )


            # ====================================================
            # FINAL MEASUREMENT IMAGE
            # ====================================================

            st.subheader(
                "Measured Lens"
            )

            st.image(
                result["measured_path"],
                caption="Detected lens dimensions",
                use_container_width=True
            )


            # ====================================================
            # PROCESSING PIPELINE
            # ====================================================

            st.subheader(
                "Computer Vision Pipeline"
            )


            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "ArUco Detection",
                    "Rectified Image",
                    "Segmentation Mask",
                    "Final Measurement"
                ]
            )


            with tab1:

                st.image(
                    result["aruco_path"],
                    caption="Detected ArUco reference markers",
                    use_container_width=True
                )


            with tab2:

                st.image(
                    result["rectified_path"],
                    caption="Perspective-corrected image",
                    use_container_width=True
                )


            with tab3:

                st.image(
                    result["mask_path"],
                    caption="SAM 2 lens segmentation",
                    use_container_width=True
                )


            with tab4:

                st.image(
                    result["measured_path"],
                    caption="Final physical measurement",
                    use_container_width=True
                )


            # ====================================================
            # TECHNICAL DETAILS
            # ====================================================

            with st.expander(
                "Technical details"
            ):

                st.write(
                    "ArUco markers detected:",
                    result["markers_detected"]
                )

                st.write(
                    "Marker IDs:",
                    result["marker_ids"]
                )

                st.write(
                    "X calibration:",
                    result["pixels_per_mm_x"],
                    "pixels/mm"
                )

                st.write(
                    "Y calibration:",
                    result["pixels_per_mm_y"],
                    "pixels/mm"
                )

                st.write(
                    "SAM candidate masks:",
                    result["sam_candidates"]
                )

                st.write(
                    "Marker padding fallback used:",
                    result["padding_used"]
                )


            # ----------------------------------------
            # Clean temporary file
            # ----------------------------------------

            if os.path.exists(
                temp_path
            ):

                os.remove(
                    temp_path
                )


        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )
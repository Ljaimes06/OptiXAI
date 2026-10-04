# OptiXAI 👓

OptiXAI is a mobile web application for measuring recycled eyeglass lenses from a smartphone photo using computer vision and AI.

## Live Demo

https://optixai-codeml.streamlit.app

## GitHub

https://github.com/Ljaimes06/OptiXAI

## What It Does

The user places an eyeglass lens inside a reference area containing four ArUco markers and takes a picture directly from the web app.

OptiXAI then:

1. Detects the four ArUco markers.
2. Corrects perspective to create a top-down image.
3. Uses the known 37.2 mm marker size to calculate pixels per millimeter.
4. Uses SAM 2.1 to segment the transparent eyeglass lens.
5. Extracts the lens contour.
6. Converts the contour into real-world millimeter coordinates.
7. Calculates:
   - A: lens width
   - B: lens height
   - Perimeter
8. Displays intermediate computer-vision results for validation.

## Capture Setup

The hackathon prototype uses four printed ArUco markers as the physical reference.

Marker size:

37.2 mm × 37.2 mm

All four markers should be visible in the photo.

The application also includes a fallback that adds temporary image padding when a marker is too close to the image boundary.

A future consumer version could use a standard bank card as the size reference instead of printed markers.

## Computer Vision Pipeline

Photo  
→ ArUco detection  
→ Perspective correction  
→ Pixel/mm calibration  
→ SAM 2.1 segmentation  
→ Lens contour  
→ Physical measurements

## AI Model

We use Meta's pretrained SAM 2.1 Hiera Tiny model through Hugging Face Transformers:

facebook/sam2.1-hiera-tiny

SAM 2 is used for zero-shot segmentation of the transparent lens.

No custom fine-tuning was performed during the hackathon.

SAM 2 model checkpoints and code are licensed under Apache 2.0.

## Validation

We tested repeated photographs of real eyeglass lenses under different positions and camera angles.

Example validation for a lens physically measured at 51 mm × 34 mm:

- 10/10 images successfully processed
- Average measured size: approximately 50.51 mm × 34.26 mm
- Average width error: 0.76 mm
- Average height error: 0.62 mm
- Worst width error: 1.51 mm
- Worst height error: 1.88 mm

A second right lens from the same pair was also tested across 10 images:

- 10/10 images successfully processed
- Average measured size: approximately 50.62 mm × 34.76 mm
- Average width error: 0.48 mm
- Average height error: 0.76 mm

## Technologies

- Python
- OpenCV
- ArUco
- PyTorch
- Hugging Face Transformers
- Meta SAM 2.1
- NumPy
- Pillow
- Streamlit
- Git / GitHub

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Ljaimes06/OptiXAI.git
cd OptiXAI
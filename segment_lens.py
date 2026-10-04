import torch
import numpy as np
from PIL import Image
from transformers import Sam2Processor, Sam2Model


# ----------------------------------------
# 1. Load pretrained SAM 2 model
# ----------------------------------------

model_name = "facebook/sam2.1-hiera-tiny"

processor = Sam2Processor.from_pretrained(model_name)
model = Sam2Model.from_pretrained(model_name)

model.eval()


# ----------------------------------------
# 2. Load rectified image
# ----------------------------------------

image_path = "output/rectified.jpg"

image = Image.open(image_path).convert("RGB")

width, height = image.size

print("Image size:", width, "x", height)


# ----------------------------------------
# 3. Bounding box around the lens
# ----------------------------------------

# Format:
# [x_min, y_min, x_max, y_max]

input_boxes = [[[
    120,
    200,
    255,
    410
]]]


# ----------------------------------------
# 4. Prepare image
# ----------------------------------------

inputs = processor(
    images=image,
    input_boxes=input_boxes,
    return_tensors="pt"
)


# ----------------------------------------
# 5. Run segmentation
# ----------------------------------------

with torch.no_grad():
    outputs = model(
        **inputs,
        multimask_output=False
    )


# ----------------------------------------
# 6. Convert prediction into mask
# ----------------------------------------

masks = processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"]
)

mask = masks[0][0][0].numpy()

mask = mask.astype(np.uint8) * 255


# ----------------------------------------
# 7. Save mask
# ----------------------------------------

mask_image = Image.fromarray(mask)

output_path = "output/lens_mask.jpg"

mask_image.save(output_path)

print("Segmentation complete")
print("Mask saved to:", output_path)
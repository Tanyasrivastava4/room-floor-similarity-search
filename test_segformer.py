from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

print("Downloading SegFormer (ADE20K scene parsing model)...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully.")

# ADE20K class list - index 3 is "floor"
print("Number of classes:", model.config.num_labels)
print("Class 3 label:", model.config.id2label[3])

# quick test on one real query image
image = Image.open("data/query/1.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # shape: (1, num_classes, H, W) - low res

print("Logits shape:", logits.shape)

# upsample logits to original image size and get predicted class per pixel
upsampled = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
pred_mask = upsampled.argmax(dim=1)[0]  # (H, W) - class id per pixel

floor_class_id = 3
floor_pixel_count = (pred_mask == floor_class_id).sum().item()
total_pixels = pred_mask.numel()
print(f"Floor pixels: {floor_pixel_count} / {total_pixels} "
      f"({floor_pixel_count/total_pixels:.2%})")

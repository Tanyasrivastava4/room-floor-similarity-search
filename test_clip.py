from transformers import CLIPModel, CLIPProcessor
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("Model loaded successfully.")

dummy_input = torch.rand(1, 3, 224, 224)

with torch.no_grad():
    vision_outputs = model.get_image_features(pixel_values=dummy_input)
    image_embeds = vision_outputs.pooler_output   # already the final 512-dim CLIP embedding

print("Final embedding shape:", image_embeds.shape)
print("First 5 values:", image_embeds[0][:5])







#from transformers import CLIPModel, CLIPProcessor
#import torch
#
#print("Downloading CLIP model (openai/clip-vit-base-patch32)...")
#model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#print("Model loaded successfully.")
#
## quick sanity check with a random tensor shaped like an image
#dummy_input = torch.rand(1, 3, 224, 224)
#with torch.no_grad():
#    features = model.get_image_features(pixel_values=dummy_input)
#
#print("Embedding shape:", features.shape)
#print("First 5 values:", features[0][:5])
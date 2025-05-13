from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Ask user for image path
path = input("Please provide the image path: ").strip()

# Open image
try:
    img = Image.open(path).convert("RGB")
except FileNotFoundError:
    print("❌ Error: File not found. Please check the path.")
    exit()
except Exception as e:
    print(f"❌ Error loading image: {e}")
    exit()

# Process and caption
data = processor(images=img, return_tensors="pt")

with torch.no_grad():
    result = model.generate(**data)

caption = processor.decode(result[0], skip_special_tokens=True)

print("✅ Image description:", caption)

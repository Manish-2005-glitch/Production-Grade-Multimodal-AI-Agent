from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def caption_image(image_np):
    image = Image.fromarray(image_np[:, :, ::-1])
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)

    return processor.decode(out[0], skip_special_tokens=True)

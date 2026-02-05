from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption(crop):
    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inputs = processor(img, return_tensors = "pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens = True)
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import gradio as gr

# Load processor and model only once, outside the function
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Create Gradio interface
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Text(label="Generated Caption"),
    title="BLIP Image Captioning",
    description="Upload an image to get an AI-generated caption using the BLIP model."
)

demo.launch()

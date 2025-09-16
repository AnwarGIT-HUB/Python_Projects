from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
#import requests
import matplotlib.pyplot as plt

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#image_url = "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg"
image_url = r"C:\Users\swaga\Downloads\istockphoto-1448152453-1024x1024.jpg"
#image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
image = Image.open(image_url).convert('RGB')

# Show the image
plt.imshow(image)
plt.axis("off")
plt.show()

inputs = processor(image, return_tensors="pt")

output = model.generate(**inputs)
caption = processor.decode(output[0], skip_special_tokens=True)

print("Generated Caption:", caption)
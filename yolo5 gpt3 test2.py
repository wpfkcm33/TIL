import io
import os
import openai
import requests
import torch
import torchvision.transforms as T

from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Replace with your OpenAI API key
openai.api_key = "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"

# Load pre-trained YOLO model (using Faster R-CNN with ResNet-50 backbone as an example)
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Download an image from the internet
image_url = "https://cdn3.incapto.com/wp-content/uploads/2020/06/Barista.jpg"
response = requests.get(image_url)
image_data = response.content

# Open the image and convert it to a tensor
image = Image.open(io.BytesIO(image_data)).convert("RGB")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image)

# Detect objects in the image
with torch.no_grad():
    predictions = model([image_tensor])

# Extract object names and their confidence scores
object_labels = predictions[0]["labels"].tolist()
object_scores = predictions[0]["scores"].tolist()

# Generate a natural language description of the objects detected
detected_objects = ", ".join([f"{score:.1%} {label}" for score, label in zip(object_scores, object_labels)])
prompt = f"Describe an image containing the following objects: {detected_objects}"
response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=50, n=1, stop=None, temperature=0.5)

# Print the generated description
print(response.choices[0].text.strip()) 
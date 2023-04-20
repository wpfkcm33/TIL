import io
import os
import openai
import requests
import torch
import torchvision.transforms as T

from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Replace with your OpenAI API key
openai.api_key = "your_key"

# Load pre-trained YOLO model (using Faster R-CNN with ResNet-50 backbone as an example)
model = fasterrcnn_resnet50_fpn(weights=True)
model.eval()

# Download an image from the internet
image_url = "https://www.yeosu.go.kr/tour/build/images/p132/p1322008/p1322008678450-1.jpg/666x1x70/666x1_p1322008678450-1.jpg"
response = requests.get(image_url)
image_data = response.content

# Open the image and convert it to a tensor
image = Image.open(io.BytesIO(image_data)).convert("RGB")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image)

# Detect objects in the image
with torch.no_grad():
    predictions = model([image_tensor])

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Extract object names and their confidence scores

object_labels = predictions[0]["labels"].tolist()
object_scores = predictions[0]["scores"].tolist()
object_boxes = predictions[0]["boxes"].tolist()
object_names = [COCO_INSTANCE_CATEGORY_NAMES[label] for label in object_labels]

top_n = 5
object_labels = object_labels[:top_n]
object_scores = object_scores[:top_n]
object_boxes = object_boxes[:top_n]


# Generate a natural language description of the objects detected
object_descriptions = []
for label, score, box in zip(object_labels, object_scores, object_boxes):
    object_descriptions.append(f"{label} with {score:.1%} confidence at ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")

detected_objects = ", ".join([f"{COCO_INSTANCE_CATEGORY_NAMES[label]} with {score:.1%} confidence at {tuple(box)}" for label, score, box in zip(object_labels, object_scores, object_boxes)])
prompt = f"Describe a scene in an image containing the following objects: {detected_objects}. Please provide a narrative of what might be happening in the image."
response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=50, n=1, stop=None, temperature=0.5)

# Print the generated description
print(response.choices[0].text.strip())
print(detected_objects)

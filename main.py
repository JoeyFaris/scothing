from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace the current model loading with a DeepFashion-trained model
model = torch.load('path_to_deepfashion_model.pth')
model.eval()

# Update clothing categories to match DeepFashion categories
clothing_categories = [
    'T-shirt', 'Dress', 'Pants', 'Skirt', 'Jacket', 'Coat', 'Shorts',
    'Sweater', 'Shirt', 'Blouse', 'Blazer', 'Cardigan', 'Jumpsuit',
    'Romper', 'Vest', 'Hoodie', 'Sweatshirt', 'Tank Top'
]
logging.info(f"Clothing categories defined: {clothing_categories}")

@app.route('/analyze_clothing', methods=['POST'])
def analyze_clothing():
    logging.info("Received request to analyze clothing")
    if 'image' not in request.files:
        logging.error("No image file provided in the request")
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    logging.info(f"Image opened and converted to RGB. Size: {image.size}")
    
    # Transform the image
    logging.info("Applying image transformations:")
    logging.info(f"  - Resizing to 256x256")
    logging.info(f"  - Center cropping to 224x224")
    logging.info(f"  - Converting to tensor")
    logging.info(f"  - Normalizing with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    logging.info(f"Image transformed. Tensor shape: {input_batch.shape}")

    # Perform inference
    logging.info("Performing model inference...")
    with torch.no_grad():
        output = model(input_batch)
    logging.info(f"Model output shape: {output.shape}")

    # Process results
    probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100
    top_prob, top_catid = torch.topk(probabilities, 3)

    detected_items = []
    for i in range(top_prob.size(0)):
        category = clothing_categories[top_catid[i]]
        confidence = float(top_prob[i])
        logging.info(f"  - {category}: {confidence:.2f}% confidence")
        if confidence > 10:  # Only consider predictions with >10% confidence
            detected_items.append({
                'category': category,
                'confidence': confidence
            })

    logging.info(f"Final detected items: {detected_items}")
    return jsonify({'detected_items': detected_items})

if __name__ == '__main__':
    logging.info("Starting the Flask application")
    app.run(debug=True)

from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import logging
from sklearn.cluster import KMeans

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

def get_dominant_colors(image, n_colors=3):
    # Resize image to speed up processing
    img = cv2.resize(image, (150, 150))
    # Reshape the image to be a list of pixels
    pixels = img.reshape(-1, 3)
    # Cluster the pixel intensities
    clt = KMeans(n_clusters=n_colors)
    clt.fit(pixels)
    # Get the colors as percentage of each cluster
    percent = []
    for i in range(len(clt.cluster_centers_)):
        j = np.where(clt.labels_ == i)[0]
        percent.append(len(j) / len(clt.labels_))
    # Get the colors in RGB
    colors = clt.cluster_centers_
    # Convert to integer RGB values
    colors = colors.astype(int)
    # Sort colors by percentage
    colors_percent = sorted(zip(colors, percent), key=lambda x: x[1], reverse=True)
    return colors_percent

@app.route('/analyze_clothing', methods=['POST'])
def analyze_clothing():
    logging.info("Received request to analyze clothing")
    if 'image' not in request.files:
        logging.error("No image file provided in the request")
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    logging.info(f"Image opened and converted to RGB. Size: {image.size}")
    
    # Transform the image for model input
    logging.info("Applying image transformations:")
    logging.info(f"  - Resizing to 256x256")
    logging.info(f"  - Center cropping to 224x224")
    logging.info(f"  - Converting to tensor")
    logging.info(f"  - Normalizing with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
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

    # Analyze colors
    logging.info("Analyzing dominant colors...")
    image_np = np.array(image)
    colors_percent = get_dominant_colors(image_np)
    color_info = []
    for color, percent in colors_percent:
        color_info.append({
            'color': color.tolist(),
            'percentage': round(percent * 100, 2)
        })
    logging.info(f"Dominant colors: {color_info}")

    result = {
        'detected_items': detected_items,
        'dominant_colors': color_info
    }

    logging.info(f"Final result: {result}")
    return jsonify(result)

if __name__ == '__main__':
    logging.info("Starting the Flask application")
    app.run(debug=True)

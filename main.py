import torch
from torchvision import transforms
from PIL import Image
from model.model import ClothingAnalyzer
from model.config import CONFIG
from api.utils import process_prediction

def main():
    # Initialize the model
    model = ClothingAnalyzer(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes']
    )
    model.eval()

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Function to analyze a single image
    def analyze_image(image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        predictions = process_prediction(output)
        return predictions

    # Example usage
    image_path = "path/to/your/image.jpg"
    results = analyze_image(image_path)
    
    print(f"Analysis results for {image_path}:")
    for category, probability in results:
        print(f"{category}: {probability:.2f}")

if __name__ == "__main__":
    main()

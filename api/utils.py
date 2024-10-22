import torch
from model.model import ClothingAnalyzer
from model.config import CONFIG

def load_model():
    """Initialize model with config settings."""
    model = ClothingAnalyzer(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes']
    )
    return model

def process_prediction(outputs, top_k=3):
    """Convert model outputs to class predictions."""
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_prob, top_class = torch.topk(probabilities, top_k)
    return [(CONFIG['classes'][idx], prob.item()) 
            for prob, idx in zip(top_prob[0], top_class[0])]
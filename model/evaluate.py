import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .model import ClothingAnalyzer

def evaluate_model(model, test_loader, criterion):
    """
    Evaluate a trained model on a test dataset
    
    Args:
        model (ClothingAnalyzer): The trained model to evaluate
        test_loader (DataLoader): DataLoader containing the test dataset
        criterion: The loss function to use for evaluation
        
    Returns:
        dict: Dictionary containing evaluation metrics (loss, accuracy, precision, recall, f1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(test_loader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

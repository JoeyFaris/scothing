import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from .model import ClothingAnalyzer
from .config import CONFIG
from .evaluate import evaluate_model

def train_model(train_loader, val_loader, epochs=10):
    """
    Train the clothing classification model
    
    Args:
        train_loader (DataLoader): DataLoader containing training data
        val_loader (DataLoader): DataLoader containing validation data 
        epochs (int): Number of training epochs
        
    Returns:
        ClothingAnalyzer: The trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and move to device
    model = ClothingAnalyzer(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes']
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
                
        # Evaluate on validation set
        val_metrics = evaluate_model(model, val_loader, criterion)
        print(f'Validation metrics - Epoch {epoch+1}:')
        print(f'Loss: {val_metrics["loss"]:.4f}, '
              f'Accuracy: {val_metrics["accuracy"]:.4f}')
    
    return model
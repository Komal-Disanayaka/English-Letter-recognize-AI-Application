import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import time
import copy
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set device - RTX 4060 with CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Only print GPU info once at startup
if torch.cuda.is_available():
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    # Enable cuDNN benchmarking for optimal performance
    torch.backends.cudnn.benchmark = True
else:
    print(f"Using device: {device} (CPU mode)")

# Data directories
data_dir = Path("Data_Set")
train_dir = data_dir / "train"
val_dir = data_dir / "validation"
test_dir = data_dir / "test"

# Hyperparameters optimized for RTX 4060 (8GB VRAM)
BATCH_SIZE = 64  # Good balance for 8GB VRAM
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
NUM_WORKERS = 4  # For efficient data loading
IMG_SIZE = 224  # ResNet-18 standard input size

# Data augmentation and preprocessing
# Training: Strong augmentation for better generalization
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),  # Rotate characters slightly
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation and Test: Only basic preprocessing
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("Loading datasets...")
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transforms)

# Create data loaders with pin_memory for faster GPU transfer
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Get number of classes
num_classes = len(train_dataset.classes)
print(f"\nDataset Information:")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes[:10]}... (showing first 10)")

# Save class mapping
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
torch.save({'class_to_idx': class_to_idx, 'idx_to_class': idx_to_class}, 'class_mapping.pth')

# Build ResNet-18 model
print("\nBuilding ResNet-18 model...")
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Modify the final layer for our number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Move model to GPU
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Early stopping parameters
    early_stopping_patience = 15
    early_stopping_counter = 0
    best_val_loss = float('inf')
    
    # For plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store metrics for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())
                
                # Learning rate scheduler step
                scheduler.step(epoch_loss)
                
                # Print learning rate when it changes
                current_lr = optimizer.param_groups[0]['lr']
                if epoch == 0 or current_lr != optimizer.param_groups[0]['lr']:
                    print(f'Learning Rate: {current_lr:.6f}')
                
                # Early stopping check
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    print(f'Early stopping counter: {early_stopping_counter}/{early_stopping_patience}')
                
                # Deep copy the model if it's the best
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'class_to_idx': class_to_idx
                    }, 'best_model.pth')
                    print(f'✓ New best model saved with accuracy: {best_acc:.4f}')
        
        # Check if early stopping should trigger
        if early_stopping_counter >= early_stopping_patience:
            print(f'\n⚠ Early stopping triggered! No improvement for {early_stopping_patience} epochs.')
            print(f'Best validation loss: {best_val_loss:.4f}')
            print(f'Best validation accuracy: {best_acc:.4f}')
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, f'checkpoint_epoch_{epoch+1}.pth')
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    return model, train_losses, val_losses, train_accs, val_accs

# Function to plot training history
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.close()

# Test the model
def test_model(model, test_loader):
    model.eval()
    running_corrects = 0
    total = 0
    
    print("\nTesting model on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    
    test_acc = running_corrects.double() / total
    print(f'Test Accuracy: {test_acc:.4f}')
    return test_acc

if __name__ == '__main__':
    print("\n" + "="*50)
    print("CHARACTER RECOGNITION SYSTEM - ResNet-18")
    print("="*50)
    
    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS
    )
    
    # Test the model
    test_acc = test_model(model, test_loader)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class
    }, 'final_model.pth')
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best model saved as 'best_model.pth'")
    print(f"Final model saved as 'final_model.pth'")
    print(f"Class mapping saved as 'class_mapping.pth'")

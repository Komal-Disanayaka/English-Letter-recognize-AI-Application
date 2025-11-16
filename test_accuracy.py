import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from pathlib import Path

def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data directories
    data_dir = Path("Data_Set")
    test_dir = data_dir / "test"

    # Parameters
    BATCH_SIZE = 64
    IMG_SIZE = 224
    NUM_WORKERS = 4

    # Test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(test_dataset.classes)}")

    # Load class mapping
    class_mapping = torch.load('class_mapping.pth')
    class_to_idx = class_mapping['class_to_idx']

    # Load best model
    print("\nLoading best_model.pth...")
    checkpoint = torch.load('best_model.pth', map_location=device)

    # Build model
    num_classes = len(class_to_idx)
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    if 'best_acc' in checkpoint:
        print(f"Best validation accuracy during training: {checkpoint['best_acc']:.4f}")

    # Test the model
    print("\n" + "="*70)
    print("TESTING MODEL ON TEST SET")
    print("="*70)

    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    test_acc = running_corrects.double() / total

    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Correct predictions: {running_corrects}/{total}")
    print("="*70)

if __name__ == '__main__':
    main()

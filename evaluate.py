import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
import json

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data directories
data_dir = Path("Data_Set")
test_dir = data_dir / "test"

# Parameters
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 4

# Test transforms (same as validation)
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
idx_to_class = class_mapping['idx_to_class']

# Convert idx_to_class keys from string to int if needed
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

# Create character labels (for better visualization)
def get_char_label(class_idx):
    """Convert class index to readable character"""
    code = int(idx_to_class[class_idx])
    if code == 999:
        return "Unknown"
    try:
        return chr(code)
    except:
        return str(code)

# Load model
def load_model(model_path='best_model.pth'):
    print(f"\nLoading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get number of classes
    num_classes = len(class_to_idx)
    
    # Build model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    if 'best_acc' in checkpoint:
        print(f"Best validation accuracy: {checkpoint['best_acc']:.4f}")
    
    return model

# Evaluate model and get predictions
def evaluate_model(model, test_loader):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    print("\nGenerating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # For large number of classes, plot a subset or simplified version
    num_classes = len(class_names)
    
    if num_classes > 30:
        # Plot normalized confusion matrix with smaller size
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                   cbar_kws={'label': 'Normalized Count'})
        plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add some class labels at intervals
        step = max(1, num_classes // 20)
        tick_positions = list(range(0, num_classes, step))
        tick_labels = [get_char_label(i) for i in tick_positions]
        plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
        plt.yticks(tick_positions, tick_labels, rotation=0)
    else:
        # Plot full confusion matrix for smaller number of classes
        plt.figure(figsize=(15, 13))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[get_char_label(i) for i in range(num_classes)],
                   yticklabels=[get_char_label(i) for i in range(num_classes)])
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as '{save_path}'")
    plt.close()

# Plot per-class accuracy
def plot_class_accuracy(y_true, y_pred, save_path='class_accuracy.png'):
    print("\nCalculating per-class accuracy...")
    
    num_classes = len(np.unique(y_true))
    class_accuracies = []
    class_labels = []
    
    for i in range(num_classes):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            class_accuracies.append(acc * 100)
            class_labels.append(get_char_label(i))
    
    # Sort by accuracy
    sorted_indices = np.argsort(class_accuracies)
    class_accuracies = [class_accuracies[i] for i in sorted_indices]
    class_labels = [class_labels[i] for i in sorted_indices]
    
    # Plot
    plt.figure(figsize=(20, 10))
    bars = plt.bar(range(len(class_accuracies)), class_accuracies, color='steelblue', alpha=0.8)
    
    # Color bars by accuracy (red for low, green for high)
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        if acc < 70:
            bar.set_color('red')
        elif acc < 85:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.xlabel('Character Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=16, pad=20)
    plt.xticks(range(len(class_labels)), class_labels, rotation=90, ha='right')
    plt.axhline(y=np.mean(class_accuracies), color='red', linestyle='--', 
               label=f'Mean Accuracy: {np.mean(class_accuracies):.2f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class accuracy plot saved as '{save_path}'")
    plt.close()
    
    return class_accuracies, class_labels

# Find worst performing classes
def analyze_worst_classes(y_true, y_pred, top_n=10):
    print(f"\nAnalyzing {top_n} worst performing classes...")
    
    num_classes = len(np.unique(y_true))
    class_accuracies = {}
    
    for i in range(num_classes):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum()
            class_accuracies[i] = {
                'accuracy': acc,
                'total': mask.sum(),
                'correct': (y_pred[mask] == i).sum(),
                'char': get_char_label(i)
            }
    
    # Sort by accuracy
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\n{top_n} Worst Performing Classes:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Class':<10} {'Char':<8} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 70)
    
    for rank, (class_idx, info) in enumerate(sorted_classes[:top_n], 1):
        print(f"{rank:<6} {class_idx:<10} {info['char']:<8} "
              f"{info['accuracy']*100:>6.2f}%     "
              f"{info['correct']}/{info['total']}")
    
    return sorted_classes[:top_n]

# Generate classification report
def generate_classification_report(y_true, y_pred):
    print("\nGenerating classification report...")
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
    
    # Precision, Recall, F1-Score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print(f"\nWeighted Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Macro average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print(f"\nMacro Average Metrics:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall: {recall_macro:.4f}")
    print(f"  F1-Score: {f1_macro:.4f}")
    
    # Save detailed report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    with open('classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\nDetailed classification report saved as 'classification_report.json'")
    
    return {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

# Main evaluation
def main():
    print("="*70)
    print("CHARACTER RECOGNITION SYSTEM - MODEL EVALUATION")
    print("="*70)
    
    # Load model
    model = load_model('best_model.pth')
    
    # Evaluate
    y_true, y_pred, y_probs = evaluate_model(model, test_loader)
    
    # Generate metrics
    metrics = generate_classification_report(y_true, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, test_dataset.classes)
    
    # Plot per-class accuracy
    class_accs, class_labels = plot_class_accuracy(y_true, y_pred)
    
    # Analyze worst performing classes
    worst_classes = analyze_worst_classes(y_true, y_pred, top_n=10)
    
    # Save evaluation results
    results = {
        'overall_metrics': metrics,
        'class_accuracies': {label: acc for label, acc in zip(class_labels, class_accs)},
        'worst_performing': [
            {
                'class_idx': int(idx),
                'character': info['char'],
                'accuracy': float(info['accuracy']),
                'correct': int(info['correct']),
                'total': int(info['total'])
            }
            for idx, info in worst_classes
        ]
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"Results saved:")
    print(f"  - confusion_matrix.png")
    print(f"  - class_accuracy.png")
    print(f"  - classification_report.json")
    print(f"  - evaluation_results.json")

if __name__ == '__main__':
    main()

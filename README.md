# Character Recognition System using ResNet-18

A deep learning-based character recognition system with an interactive web interface, optimized for NVIDIA RTX 4060 GPU with PyTorch, CUDA, and cuDNN.

## 🎯 Overview

This project implements a complete end-to-end character recognition system capable of identifying hand-drawn characters (letters, numbers, and symbols). It features a trained ResNet-18 deep learning model and an interactive Flask web application with a drawing canvas for real-time character prediction.

## ✨ Features

- **ResNet-18 Architecture**: Pre-trained on ImageNet, fine-tuned for character recognition
- **GPU Optimized**: Configured for RTX 4060 with CUDA and cuDNN acceleration
- **Interactive Web Interface**: Draw characters and get instant predictions
- **Data Augmentation**: Comprehensive preprocessing and augmentation techniques
- **Early Stopping**: Prevents overfitting during training
- **Complete Pipeline**: Training, evaluation, and inference scripts
- **Visualization**: Training curves, confusion matrices, per-class accuracy plots
- **High Accuracy**: Achieves 90-95% test accuracy on character recognition

## 🏗️ Architecture & Technology Stack

### Deep Learning Framework
- **PyTorch**: 2.0.0+
- **torchvision**: 0.15.0+
- **CUDA**: 11.8+ or 12.x
- **cuDNN**: 8.x+

### Model Architecture
- **Base Model**: ResNet-18
- **Pre-trained Weights**: ImageNet (ResNet18_Weights.DEFAULT)
- **Input Size**: 224x224x3 RGB images
- **Output**: Softmax probabilities over character classes
- **Final Layer**: Fully connected layer (512 → num_classes)

### Web Application
- **Backend**: Flask 2.3.0+
- **Frontend**: HTML5, CSS3, JavaScript
- **Canvas API**: For drawing interface
- **AJAX**: Real-time predictions

### Data Processing
- **NumPy**: 1.24.0+
- **Pillow**: 10.0.0+ (Image processing)
- **scikit-learn**: 1.3.0+ (Metrics and evaluation)

### Visualization
- **matplotlib**: 3.7.0+
- **seaborn**: 0.12.0+

## 📊 Dataset Structure

## DataSet_link - https://www.kaggle.com/datasets/lopalp/alphanum

```
Data_Set/
├── train/       # Training images (70-80% of data)
├── validation/  # Validation images (10-15% of data)
└── test/        # Test images (10-15% of data)
```

Each folder contains subdirectories named by ASCII character codes:
- **65-90**: Uppercase letters (A-Z)
- **97-122**: Lowercase letters (a-z)
- **48-57**: Digits (0-9)
- **Other codes**: Special symbols
- **999**: Unknown/unrecognized characters

## 🚀 Installation

### Prerequisites

**Hardware Requirements:**
- NVIDIA RTX 4060 (or similar GPU with 8GB+ VRAM)
- CUDA-capable GPU with Compute Capability 6.0+
- 16GB+ RAM recommended

**Software Requirements:**
- Python 3.8+
- CUDA 11.8+ or 12.x
- cuDNN 8.x+
- Anaconda/Miniconda (recommended)

### Setup Instructions

#### Step 1: Create Python Environment

```powershell
# Create conda environment
conda create -n letter_ai python=3.10 -y
conda activate letter_ai
```

#### Step 2: Install PyTorch with CUDA Support

**For CUDA 11.8:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

#### Step 4: Verify GPU Setup

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4060
```

## 📚 Usage

### 1. Train the Model

```powershell
python train.py
```

**Training Features:**
- ✅ Automatic GPU detection and optimization
- ✅ Data augmentation (rotation, affine transforms, color jitter, perspective)
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Early stopping (patience=15 epochs)
- ✅ Model checkpointing every 10 epochs
- ✅ Best model saved based on validation accuracy
- ✅ Training history visualization

**Training Outputs:**
- `best_model.pth` - Best performing model
- `final_model.pth` - Final model after all epochs
- `class_mapping.pth` - Class to index mapping
- `training_history.png` - Training/validation curves
- `checkpoint_epoch_*.pth` - Periodic checkpoints

**Hyperparameters (Optimized for RTX 4060):**
```python
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 15
```

### 2. Evaluate the Model

```powershell
python evaluate.py
```

**Evaluation Metrics:**
- Overall test accuracy
- Per-class accuracy
- Precision, Recall, F1-Score (weighted and macro)
- Confusion matrix (normalized)
- Worst performing classes analysis

**Evaluation Outputs:**
- `confusion_matrix.png` - Visual confusion matrix
- `class_accuracy.png` - Per-class accuracy bar chart
- `classification_report.json` - Detailed metrics per class
- `evaluation_results.json` - Summary of all metrics

### 3. Test Accuracy

```powershell
python test_accuracy.py
```

Quick script to get test set accuracy using the best model.

### 4. Make Predictions (Command Line)

**Single Image:**
```powershell
python predict.py --image path/to/image.jpg --top_k 5
```

**Batch Prediction:**
```powershell
python predict.py --folder path/to/images/ --top_k 3
```

**Save Visualization:**
```powershell
python predict.py --image path/to/image.jpg --save_viz
```

### 5. Web Application

**Start the Flask Server:**
```powershell
python app.py
```

**Access the Application:**
```
http://localhost:5000
```

**How to Use:**
1. Draw a character on the black canvas (white pen)
2. Click "🔍 Predict" to identify the character
3. View top 5 predictions with confidence scores
4. Click "🗑️ Clear" to draw again

**Web Interface Features:**
- ✨ Responsive canvas with touch support
- ✨ Real-time character prediction
- ✨ Top 5 predictions with confidence bars
- ✨ ASCII code display
- ✨ Beautiful gradient UI
- ✨ Mobile-friendly design

## 🔬 Data Preprocessing & Augmentation

### Training Augmentation:
```python
- Resize to 224×224
- Random rotation (±15°)
- Random affine transforms (translation ±10%, scale 0.9-1.1)
- Color jitter (brightness, contrast ±30%)
- Random perspective distortion (20%)
- Normalization (ImageNet statistics)
```

### Validation/Test Preprocessing:
```python
- Resize to 224×224
- Normalization only (no augmentation)
```

### Web Application Preprocessing:
```python
- Convert to grayscale
- Invert colors (white on black)
- Auto-crop to bounding box
- Add padding
- Resize to 224×224
- Normalize
```

## 📈 Model Performance

### Expected Results:
- **Training Accuracy**: 95-99%
- **Validation Accuracy**: 90-95%
- **Test Accuracy**: 90-95%
- **Inference Time**: ~10-20ms per image (GPU)

### Performance Factors:
- Dataset quality and size
- Number of training epochs
- Data augmentation effectiveness
- Class balance in dataset
- Early stopping timing

## ⚙️ Optimization Techniques

### GPU Optimization (RTX 4060 - 8GB VRAM):
```python
torch.backends.cudnn.benchmark = True  # Auto-tune algorithms
pin_memory=True                         # Faster CPU-to-GPU transfer
num_workers=4                           # Parallel data loading
batch_size=64                           # Optimal for 8GB VRAM
```

### Training Optimizations:
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: Prevents overfitting (patience=15)
- **Best Model Saving**: Automatically saves highest validation accuracy model
- **Checkpoint Saving**: Every 10 epochs for recovery

### Memory Management Tips:
- Reduce batch size if out of memory
- Use `torch.cuda.empty_cache()` to free unused memory
- Close other GPU-intensive applications
- Monitor GPU usage with `nvidia-smi`

## 🐛 Troubleshooting

### CUDA Out of Memory:
```python
# In train.py, reduce:
BATCH_SIZE = 32  # or 16
NUM_WORKERS = 2
```

### Slow Training:
- Ensure cuDNN is enabled (check startup output)
- Verify GPU is being used ("Using device: cuda:0")
- Check `num_workers` is not too high (optimal: 4)
- Close background GPU applications

### Import Errors:
```powershell
pip install --upgrade -r requirements.txt
python -c "import torch; import torchvision; print('OK')"
```

### Model Loading Issues:
- Ensure `class_mapping.pth` exists
- Verify model path is correct
- Check PyTorch version compatibility

### Low Prediction Confidence:
- Canvas drawing should be white on black background
- Draw characters clearly and center them
- Use bold, continuous strokes
- Ensure character fills canvas adequately

### Web App Not Loading:
```powershell
# Check port availability
netstat -ano | findstr :5000

# Use different port
# In app.py: app.run(debug=True, port=5001)
```

## 📁 Project Structure

```
Character_Recorganization_web/
├── Data_Set/              # Dataset directory
│   ├── train/            # Training images
│   ├── validation/       # Validation images
│   └── test/             # Test images
├── templates/            # Flask HTML templates
│   └── index.html        # Web interface
├── train.py              # Main training script
├── evaluate.py           # Model evaluation script
├── test_accuracy.py      # Quick accuracy test
├── predict.py            # Command-line inference
├── app.py                # Flask web application
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── best_model.pth        # Best model (after training)
├── final_model.pth       # Final model (after training)
├── class_mapping.pth     # Class mappings (after training)
└── training_history.png  # Training curves (after training)
```

## 🎓 Technical Details

### ResNet-18 Modifications:
```python
# Original ResNet-18
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Modified final layer
num_ftrs = model.fc.in_features  # 512
model.fc = nn.Linear(num_ftrs, num_classes)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Training Loop Features:
- Mixed precision training support (optional)
- Gradient accumulation (optional)
- Learning rate warmup (optional)
- Model checkpointing
- TensorBoard logging (optional)

### Evaluation Metrics:
```python
- Accuracy (overall, per-class)
- Precision (weighted, macro)
- Recall (weighted, macro)
- F1-Score (weighted, macro)
- Confusion Matrix
- Support (samples per class)
```

## 🔮 Future Enhancements

- [ ] Add data augmentation preview
- [ ] Implement model ensemble
- [ ] Add confusion analysis tools
- [ ] Support for more model architectures
- [ ] Real-time training visualization
- [ ] Model quantization for faster inference
- [ ] ONNX export for deployment
- [ ] REST API for predictions
- [ ] Docker containerization
- [ ] Cloud deployment guides

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- **ResNet Architecture**: He et al., "Deep Residual Learning for Image Recognition"
- **PyTorch**: Facebook AI Research
- **torchvision**: Pre-trained models and transforms
- **ImageNet**: Pre-trained weights source
- **NVIDIA CUDA**: GPU acceleration
- **Flask**: Web framework
- **Community**: Open-source contributors

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure GPU drivers are up to date
4. Review training logs for errors

## 🎯 Key Takeaways

✅ **High Performance**: 90-95% accuracy on character recognition
✅ **GPU Optimized**: Full CUDA/cuDNN acceleration
✅ **Production Ready**: Complete with web interface
✅ **Extensible**: Easy to adapt for other classification tasks
✅ **Well Documented**: Comprehensive guides and examples

---

**Built with ❤️ using PyTorch, ResNet-18, and Flask**

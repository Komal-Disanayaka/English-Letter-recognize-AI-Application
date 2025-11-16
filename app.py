from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image preprocessing
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model and class mapping
print("Loading model...")
class_mapping = torch.load('class_mapping.pth', map_location=device)
class_to_idx = class_mapping['class_to_idx']
idx_to_class = class_mapping['idx_to_class']
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

checkpoint = torch.load('best_model.pth', map_location=device)

num_classes = len(class_to_idx)
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print("Model loaded successfully!")

def get_char_label(class_code):
    """Convert class code to readable character"""
    code = int(class_code)
    if code == 999:
        return "Unknown"
    try:
        return chr(code)
    except:
        return str(code)

def predict_character(image_data):
    """Predict character from image data"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert to grayscale for better preprocessing
        image_gray = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image_gray)
        
        # Invert colors - canvas draws black on white, but model expects white on black
        img_array = 255 - img_array
        
        # Find bounding box of the drawn character
        rows = np.any(img_array > 30, axis=1)
        cols = np.any(img_array > 30, axis=0)
        
        if rows.any() and cols.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add padding
            padding = 20
            rmin = max(0, rmin - padding)
            rmax = min(img_array.shape[0], rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(img_array.shape[1], cmax + padding)
            
            # Crop to bounding box
            img_array = img_array[rmin:rmax, cmin:cmax]
        
        # Convert back to PIL Image
        image_processed = Image.fromarray(img_array).convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image_processed).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, 5)
        
        # Get predictions
        predictions = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            class_code = idx_to_class[idx]
            char = get_char_label(class_code)
            predictions.append({
                'character': char,
                'confidence': float(prob * 100),
                'code': int(class_code)
            })
        
        return predictions
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        predictions = predict_character(image_data)
        
        if predictions:
            return jsonify({'predictions': predictions})
        else:
            return jsonify({'error': 'Failed to process image'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Handle clear canvas request"""
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

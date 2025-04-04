# Install required packages
!pip install transformers datasets matplotlib tqdm

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from google.colab import drive
import numpy as np
import time

# Mount Google Drive to save models and upload datasets
drive.mount('/content/drive')

# Configuration
MODEL_NAME = "google/efficientnet-b0"
BATCH_SIZE = 64  # Larger batch size for Colab GPU
NUM_EPOCHS = 15
LEARNING_RATE = 5e-5
OUTPUT_DIR = "/content/drive/MyDrive/plant_efficientnet_model"
IMAGE_SIZE = 224

# For Google Colab dataset handling
# Option 1: Upload via file uploader
from google.colab import files
print("Please upload your zipped dataset (if not already in Google Drive)")
try:
    uploaded = files.upload()
    if uploaded:
        !unzip -q {list(uploaded.keys())[0]} -d /content/plant_dataset
        DATASET_PATH = "/content/plant_dataset"
    else:
        # Option 2: Use dataset from Google Drive
        DATASET_PATH = "/content/drive/MyDrive/path_to_your_plant_dataset"  # Update this path
except:
    print("Using dataset from Drive instead")
    DATASET_PATH = "/content/drive/MyDrive/path_to_your_plant_dataset"  # Update this path

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations with augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and split dataset
print(f"Loading dataset from {DATASET_PATH}")
try:
    full_dataset = datasets.ImageFolder(root=DATASET_PATH)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Split the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use different transforms for train and validation
    train_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=val_transform)
    
    # Create proper splits
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    
    # Use subset instead of random_split for more control
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    raise

# Load model and processor from Hugging Face
print(f"Loading EfficientNet model: {MODEL_NAME}")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)

# Save class mapping
with open(os.path.join(OUTPUT_DIR, "class_mapping.txt"), "w") as f:
    for i, class_name in enumerate(class_names):
        f.write(f"{i}: {class_name}\n")

# Move model to device
model = model.to(device)

# Set up optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training metrics tracking
train_losses = []
val_losses = []
val_accuracies = []
best_val_accuracy = 0.0

# Training loop
print("Starting training...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    
    for images, labels in train_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        train_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]")
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            
            # Statistics
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            val_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    val_accuracies.append(accuracy)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Save the best model
    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        # Save with Hugging Face format
        model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        # Also save PyTorch format
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_pytorch.pth"))
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

total_time = time.time() - start_time
print(f"Training completed in {total_time / 60:.2f} minutes")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Validation Accuracy')

plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
plt.show()

# Example inference function
def predict_image(image_path):
    from PIL import Image
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    # Get predicted class and probabilities
    probs = torch.nn.functional.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    pred_prob = probs[0][pred_class].item()
    
    return {
        'class_name': class_names[pred_class],
        'class_id': pred_class,
        'probability': f"{pred_prob:.4f}"
    }

print("Sample usage for inference:")
print("""
# To use your trained model for inference:
from PIL import Image

# Load an image
img = Image.open('path_to_test_image.jpg')

# Display the image
plt.imshow(img)
plt.axis('off')
plt.show()

# Get prediction
result = predict_image('path_to_test_image.jpg')
print(f"Prediction: {result['class_name']} (Confidence: {result['probability']})")
""")
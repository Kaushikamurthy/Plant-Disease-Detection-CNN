import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np

class AdvancedPlantClassifier(nn.Module):
    def __init__(self, num_classes=38):
        super(AdvancedPlantClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdvancedPlantClassifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    return image

def get_prediction(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities[0]

def main():
    st.set_page_config(page_title="Plant Disease Detection", layout="wide")
    
    st.sidebar.title("ℹ️ About the Project")
    st.sidebar.markdown(
        """
        **Project Scope:**
        - Develop an end-to-end plant disease detection system.
        - Provide a tool for farmers to quickly diagnose plant diseases.
        - Implement a CNN model for classification and compare its performance with pre-trained models.
        """
    )
    
    st.sidebar.title("📌 How to Use")
    st.sidebar.markdown(
        """
        1. Upload a clear image of a plant leaf.
        2. Click the "Detect Disease" button.
        3. View the results and predictions.
        """
    )
    
    st.sidebar.title("🌱 Supported Plants")
    st.sidebar.markdown("""
    - Apple
    - Blueberry
    - Cherry
    - Corn (Maize)
    - Grape
    - Orange
    - Peach
    - Pepper
    - Potato
    - Raspberry
    - Soybean
    - Squash
    - Strawberry
    - Tomato
    """)
    
    st.title("🌿 Plant Disease Detection System")
    st.write("Upload a leaf image to detect plant diseases")
    
    try:
        model = load_model(r"C:\Users\user\Desktop\guvi\plant disease\plant_disease_model_acc_0.9352.pth")
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Disease"):
            with st.spinner("Analyzing image..."):
                image_tensor = process_image(image)
                prediction_idx, probabilities = get_prediction(model, image_tensor)
                
                main_prediction = CLASS_NAMES[prediction_idx]
                plant_name, condition = main_prediction.split('___')
                
                st.subheader("Detection Results")
                st.markdown(f"**Plant:** {plant_name}")
                st.markdown(f"**Condition:** {condition.replace('_', ' ')}")
                st.markdown(f"**Confidence:** {probabilities[prediction_idx]*100:.2f}%")
                
                if "healthy" in condition.lower():
                    st.success("✅ Plant appears to be healthy!")
                else:
                    st.warning("⚠️ Plant may have a disease")

if __name__ == "__main__":
    main()





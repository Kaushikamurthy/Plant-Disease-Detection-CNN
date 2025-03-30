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
    model = AdvancedPlantClassifier(num_classes=38)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def get_prediction(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    return predicted.item(), probabilities[0]

translations = {
    "English": {
        "about": "**Project Scope:**\n- Develop an end-to-end plant disease detection system.\n- Provide a tool for farmers to quickly diagnose plant diseases.\n- Implement a CNN model for classification and compare its performance with pre-trained models.",
        "usage": "**How to Use**\n1. Upload a clear image of a plant leaf.\n2. Click the 'Detect Disease' button.\n3. View the results and predictions.",
        "plants": "**Supported Plants**\n- Apple\n- Blueberry\n- Cherry\n- Corn (Maize)\n- Grape\n- Orange\n- Peach\n- Pepper\n- Potato\n- Raspberry\n- Soybean\n- Squash\n- Strawberry\n- Tomato",
        "title": "🌿 Plant Disease Detection System",
        "write": "Upload a leaf image to detect plant diseases",
        "model_success": "✅ Model loaded successfully!",
        "choose_image": "Choose an image...",
        "detect_button": "Detect Disease",
        "detection_results": "Detection Results",
        "healthy_message": "✅ Plant appears to be healthy!",
        "disease_warning": "⚠️ Plant may have a disease."
    },
    "Tamil": {
        "about": "**திட்டம்:**\n- முழுமையான தாவர நோய் கண்டறியும் அமைப்பை உருவாக்குதல்.\n- விவசாயிகள் விரைவில் நோய்களை கண்டறிய உதவுதல்.\n- வகைப்படுத்த மற்றும் செயல்திறனை ஒப்பிட CNN மாதிரி பயன்படுத்துதல்.",
        "usage": "**எப்படி பயன்படுத்துவது**\n1. தாவர இலை படத்தை பதிவேற்றவும்.\n2. 'நோயைக் கண்டறி' பொத்தானை அழுத்தவும்.\n3. முடிவுகளை பார்க்கவும்.",
        "plants": "**ஆதரிக்கப்படும் தாவரங்கள்**\n- ஆப்பிள்\n- புளூபெரி\n- செர்ரி\n- மக்காச்சோளம்\n- திராட்சை\n- ஆரஞ்சு\n- பீச்\n- மிளகு\n- உருளைக்கிழங்கு\n- ராஸ்பெரி\n- சோயாபீன்\n- ஸ்க்வாஷ்\n- ஸ்ட்ராபெரி\n- தக்காளி",
        "title": "🌿 தாவர நோய் கண்டறிதல் அமைப்பு",
        "write": "தாவர நோய்களை கண்டறிய ஒரு இலையின் படத்தை பதிவேற்றவும்",
        "model_success": "✅ மாதிரி வெற்றிகரமாக ஏற்றப்பட்டது!",
        "choose_image": "ஒரு படத்தை தேர்ந்தெடுக்கவும்...",
        "detect_button": "நோயைக் கண்டறி",
        "detection_results": "கண்டறிந்த முடிவுகள்",
        "healthy_message": "✅ தாவரம் ஆரோக்கியமாக இருக்கிறது!",
        "disease_warning": "⚠️ தாவரத்தில் நோய் இருக்கலாம்."
    },
    "Kannada": {
        "about": "**ಯೋಜನೆಯ ವ್ಯಾಪ್ತಿ:**\n- ಸಂಪೂರ್ಣ ಸಸ್ಯ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ ಅಭಿವೃದ್ಧಿಪಡಿಸಿ.\n- ರೈತರು ಬೇಗನೆ ರೋಗಗಳನ್ನು ಗುರುತಿಸಲು ಸಹಾಯ ಮಾಡಿ.\n- CNN ಮಾದರಿಯನ್ನು ಜಾಗತಿಕ ಮಾದರಿಗಳೊಂದಿಗೆ ಹೋಲಿಸಿ.",
        "usage": "**ಹೆಚ್ಚು ಬಳಸುವುದು**\n1. ಸಸ್ಯ ಎಲೆಯ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ.\n2. 'ರೋಗ ಪತ್ತೆ' ಬಟನ್ ಒತ್ತಿರಿ.\n3. ಫಲಿತಾಂಶಗಳನ್ನು ನೋಡಿ.",
        "plants": "**ಬೆಂಬಲಿತ ಸಸ್ಯಗಳು**\n- ಆಪಲ್\n- ಬ್ಲೂಬೆರಿ\n- ಚೆರಿ\n- ಜೋಳ\n- ದ್ರಾಕ್ಷಿ\n- ಕಿತ್ತಳೆ\n- ಪೀಚ್\n- ಮೆಣಸು\n- ಆಲೂಗಡ್ಡೆ\n- ಹಿಂಬೆರ್ರಿ\n- ಸೋಯಾಬೀನ್\n- ಸ್ಕ್ವಾಶ್\n- ಸ್ಟ್ರಾಬೆರಿ\n- ಟೊಮ್ಯಾಟೋ",
        "title": "🌿 ಸಸ್ಯ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ",
        "write": "ಸಸ್ಯ ರೋಗ ಪತ್ತೆ ಮಾಡಲು ಎಲೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "model_success": "✅ ಮಾದರಿ ಯಶಸ್ವಿಯಾಗಿ ಲೋಡ್ ಮಾಡಲಾಗಿದೆ!",
        "choose_image": "ಚಿತ್ರವನ್ನು ಆಯ್ಕೆಮಾಡಿ...",
        "detect_button": "ರೋಗ ಪತ್ತೆ",
        "detection_results": "ಪತ್ತೆಯ ಫಲಿತಾಂಶಗಳು",
        "healthy_message": "✅ ಸಸ್ಯ ಆರೋಗ್ಯಕರವಾಗಿದೆ!",
        "disease_warning": "⚠️ ಸಸ್ಯಕ್ಕೆ ರೋಗ ಇದ್ದಿರಬಹುದು."
    },
    "Hindi": {
        "about": "**परियोजना का दायरा:**\n- पौधों के रोगों का संपूर्ण पहचान प्रणाली विकसित करना।\n- किसानों को शीघ्र निदान में सहायता करना।\n- CNN मॉडल का उपयोग करके वर्गीकरण करना।",
        "usage": "**कैसे उपयोग करें**\n1. पौधे की पत्ती की स्पष्ट छवि अपलोड करें।\n2. 'रोग पहचानें' बटन दबाएं।\n3. परिणाम देखें।",
        "plants": "**समर्थित पौधे**\n- सेब\n- ब्लूबेरी\n- चेरी\n- मकई\n- अंगूर\n- संतरा\n- आड़ू\n- मिर्च\n- आलू\n- रास्पबेरी\n- सोयाबीन\n- स्क्वैश\n- स्ट्रॉबेरी\n- टमाटर",
        "title": "🌿 पौधा रोग पहचान प्रणाली",
        "write": "पौधे की बीमारी का पता लगाने के लिए पत्ते की छवि अपलोड करें",
        "model_success": "✅ मॉडल सफलतापूर्वक लोड हुआ!",
        "choose_image": "एक छवि चुनें...",
        "detect_button": "रोग पहचानें",
        "detection_results": "पहचान परिणाम",
        "healthy_message": "✅ पौधा स्वस्थ दिख रहा है!",
        "disease_warning": "⚠️ पौधे में बीमारी हो सकती है।"
    }
}


def main():
    st.set_page_config(page_title="Plant Disease Detection", layout="wide")
    
    language = st.sidebar.radio("Select Language / மொழியை தேர்வு செய்யவும் / ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ / भाषा चुनें:", list(translations.keys()))
    
    st.sidebar.title("ℹ️ About the Project")
    st.sidebar.markdown(translations[language]["about"])
    
    st.sidebar.title("📌 How to Use")
    st.sidebar.markdown(translations[language]["usage"])
    
    st.sidebar.title("🌱 Supported Plants")
    st.sidebar.markdown(translations[language]["plants"])
    
    st.title(translations[language]["title"])
    st.write(translations[language]["write"])
    
    try:
        model = load_model(r"C:\Users\user\Desktop\guvi\plant disease\plant_disease_model_acc_0.9335.pth")
        st.success(translations[language]["model_success"])
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    uploaded_file = st.file_uploader(translations[language]["choose_image"], type=["jpg", "jpeg", "png"])
    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button(translations[language]["detect_button"]):
            with st.spinner("Analyzing image..."):
                image_tensor = process_image(image)
                prediction_idx, probabilities = get_prediction(model, image_tensor)

                main_prediction = CLASS_NAMES[prediction_idx]
                plant_name, condition = main_prediction.split('___')

                st.subheader(translations[language]["detection_results"])
                st.markdown(f"**Plant (தாவரம் | ಸಸ್ಯ | पौधा):** {plant_name}")
                st.markdown(f"**Condition (நிலை | ಪರಿಸ್ಥಿತಿ | स्थिति):** {condition.replace('_', ' ')}")
                st.markdown(f"**Confidence (நம்பிக்கை | ನಂಬಿಕೆ | विश्वास):** {probabilities[prediction_idx]*100:.2f}%")


                if "healthy" in condition.lower():
                    st.success(translations[language]["healthy_message"])
                else:
                    st.warning(translations[language]["disease_warning"])
    
if __name__ == "__main__":
    main()
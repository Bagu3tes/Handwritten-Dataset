import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
import cv2


# 1. Definition of the architecture (must match the trained model)

class SimpleCNNImproved(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. Load the trained model file (.pth)
@st.cache_resource
def load_model():
    model = SimpleCNNImproved()
    model.load_state_dict(torch.load('best_score_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# List of classes (0-9, A-Z, a-z) (ajust according to your training)
classes = [str(i) for i in range(10)] + [chr(i) for i in range(65,91)] + [chr(i) for i in range(97,123)]

# 3. Interface with Streamlit
st.title("Recognition of Handwritten Letters and Digits")

# Canvas Design
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Prever"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        img = img.convert('L')                  # Grayscale
        img = ImageOps.invert(img)              # Inverts background color black-white
        img = img.resize((28, 28))              # Resize 28x28
        img_array = np.array(img) / 255.0       
        img_array = img_array.reshape(1, 1, 28, 28)  
        tensor = torch.from_numpy(img_array).float()

        # Predição
        with torch.no_grad():
            output = model(tensor)
            _, predicted = torch.max(output, 1)
            pred_class = classes[predicted.item()]

        st.write(f"### Predição: **{pred_class}**")
        st.image(img, width=200, caption="Image size (28x28)")
    else:
        st.write("do something on the canvas")

st.write("Model trained with ~62% accuracy")
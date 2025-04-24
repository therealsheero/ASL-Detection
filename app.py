import streamlit as st
st.set_page_config(page_title="ASL Recognition", layout="wide")
# Custom CSS to inject
def set_bg_color():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #25344f;  
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Set custom CSS for the top bar
st.markdown(
    """
    <style>
    [data-testid="stHeader"] {
        background-color: #82781a !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your app content here
st.title("My App")

set_bg_color()


from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import streamlit.components.v1 as components





# -------------------- Model Setup --------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 36)
    model.load_state_dict(torch.load("asl_mobilenetv2_best.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device


model, device = load_model()

# -------------------- Labels & Transforms --------------------
labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]  # 0-9 + A-Z


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])




# -------------------- Video Frame Processor --------------------
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        hands, img = self.detector.findHands(img)

        if hands:
            x, y, w, h = hands[0]['bbox']
            offset = 20
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
                imgGray = cv2.equalizeHist(imgGray)
                pil_img = Image.fromarray(imgGray)
                img_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    sign = labels[predicted.item()]
                    conf = confidence.item()


                label_y = y - 10 if y - 30 > 10 else y + h + 30
                cv2.putText(
                    img,
                    f"{sign} ({conf:.2f})",
                    # (x, y - 30),
                    (x,label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img


# # -------------------- Streamlit UI --------------------
# st.title("🤟 ASL Real-Time Recognition")
# st.markdown("Place your hand showing a gesture within the camera frame. Press `Stop` to end webcam.")

# webrtc_streamer(
#     key="asl",
#     video_transformer_factory=ASLTransformer,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True
# )


# -------------------- Streamlit UI --------------------
st.title("🤟 ASL Real-Time Recognition")


# Create two columns (60% for webcam, 40% for ASL chart)
col1, col2 = st.columns([0.5, 0.5])


with col1:
    # st.markdown("### Live Camera Feed")
    st.markdown("### LIVE CAMERA FEED")
    webrtc_streamer(
        key="asl",
        video_transformer_factory=ASLTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
        # video_html_attrs=("autoplay", "muted", "playsinline", "width=800", "height=600")
    )

with col2:
    st.markdown("### ASL Reference Chart")
    # Replace with your actual image path
    asl_chart = Image.open("asl_alphabets.jpg")  
    # st.image(asl_chart, caption="American Sign Language Alphabet", use_column_width=True)
    st.image(asl_chart, caption="American Sign Language Alphabet", width=600)
    
    # Optional: Add legend
    st.markdown("""
    **Legend:**
    - Letters A-Z
    - Learn American Sign Language!
    """)

# Footer
st.markdown("---")
st.caption("Tip: Ensure good lighting and clear hand gestures for best results")


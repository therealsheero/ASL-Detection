import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from cvzone.HandTrackingModule import HandDetector

# -------------------- Basic Page Setup --------------------
st.set_page_config(page_title="ASL Crossword", layout="wide")

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

# Set top bar color
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

st.title("🧩 ASL Crossword Puzzle Game")

set_bg_color()

# -------------------- Load Model --------------------
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

labels = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------- Video Transformer --------------------
class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector(maxHands=1)
        self.current_sign = ""

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

                self.current_sign = sign

                label_y = y - 10 if y - 30 > 10 else y + h + 30
                cv2.putText(
                    img,
                    f"{sign} ({conf:.2f})",
                    (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

# -------------------- Crossword Board Setup --------------------
rows, cols = 5, 5

if "board" not in st.session_state:
    st.session_state.board = [["" for _ in range(cols)] for _ in range(rows)]

if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = None

if "current_letter" not in st.session_state:
    st.session_state.current_letter = ""

st.header("📋 Crossword Board")

# Display the crossword board
for row_idx in range(rows):
    col_objs = st.columns(cols)
    for col_idx in range(cols):
        cell_value = st.session_state.board[row_idx][col_idx]
        if col_objs[col_idx].button(cell_value if cell_value else " ", key=f"btn-{row_idx}-{col_idx}"):
            st.session_state.selected_cell = (row_idx, col_idx)

# -------------------- Webcam for selected cell --------------------
if st.session_state.selected_cell:
    st.success(f"Selected Cell: {st.session_state.selected_cell}")
    webrtc_ctx = webrtc_streamer(
        key=f"asl-crossword-{st.session_state.selected_cell}",
        video_transformer_factory=ASLTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_transformer:
        predicted_sign = webrtc_ctx.video_transformer.current_sign
        if predicted_sign:
            st.session_state.current_letter = predicted_sign
            st.info(f"Detected Letter: {predicted_sign}")

    if st.button("✅ Confirm Letter"):
        row, col = st.session_state.selected_cell
        st.session_state.board[row][col] = st.session_state.current_letter
        st.session_state.selected_cell = None
        st.session_state.current_letter = ""

# -------------------- Crossword Hints --------------------
# st.header("🧩 Crossword Hints")

# st.markdown("""
# ### Across:
# - 1 Across (0,0): First letter of alphabet
# - 3 Across (2,0): Number after two
# - 5 Across (4,0): Color of the sun

# ### Down:
# - 1 Down (0,0): Not off
# - 2 Down (0,2): Opposite of cold
# """)

# # -------------------- Display crossword and ASL chart side by side --------------------

# # Create two columns: 65% width for crossword, 35% width for ASL Chart
# col1, col2 = st.columns([0.65, 0.35])

# with col1:
#     st.header("🧩 Crossword Hints")

#     st.markdown("""
#     ### Across:
#     - 1 Across (0,0): First letter of alphabet
#     - 3 Across (2,0): Number after two
#     - 5 Across (4,0): Color of the sun

#     ### Down:
#     - 1 Down (0,0): Not off
#     - 2 Down (0,2): Opposite of cold
#     """)
# with col2:
#     # 🖼️ ASL Chart display
#     st.subheader("🤟 ASL Alphabet Reference")

#     from PIL import Image
#     asl_chart = Image.open("asl_alphabets.jpg")   # <--- Your ASL Chart file here
#     st.image(asl_chart, caption="American Sign Language Alphabets", use_container_width=True)

#     st.caption("Use this reference to make correct hand signs!")


# -------------------- Display crossword and ASL chart side by side --------------------

# Create two columns: 65% width for crossword, 35% width for ASL Chart
col1, col2 = st.columns([0.65, 0.35])

with col1:
    st.header("🧩 Crossword Hints")

    # Use HTML for better font size
    st.markdown("<h3>Across:</h3>", unsafe_allow_html=True)
    st.markdown("""
    <ul style="font-size:24px;">
        <li><b>1 Across (0,0):</b> First letter of alphabet</li>
        <li><b>3 Across (2,0):</b> Number after two</li>
        <li><b>5 Across (4,0):</b> Color of the sun</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Down:</h3>", unsafe_allow_html=True)
    st.markdown("""
    <ul style="font-size:24px;">
        <li><b>1 Down (0,0):</b> Not off</li>
        <li><b>2 Down (0,2):</b> Opposite of cold</li>
    </ul>
    """, unsafe_allow_html=True)

with col2:
    # 🖼️ ASL Chart display
    st.subheader("🤟 ASL Alphabet Reference")

    from PIL import Image
    asl_chart = Image.open("asl_alphabets.jpg")   # <--- Your ASL Chart file here
    st.image(asl_chart, caption="American Sign Language Alphabets", use_container_width=True)

    # st.caption("Use this reference to make correct hand signs!")
    # Instead of st.caption, use custom markdown
st.markdown(
    """
    <p style='font-size:18px; text-align:center; color:gray;'>
        Use this reference to make correct hand signs!
    </p>
    """, unsafe_allow_html=True
)



# Footer
st.markdown("---")
st.caption("Tip: Make clear gestures! ✋🏻 Good lighting helps recognition. 🚀")

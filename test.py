import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image  # Required for transforms
from cvzone.HandTrackingModule import HandDetector

# Define the model architecture (same as training)
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 36)  # Assuming 36 classes (A-Z + 0-9)

# Load the trained weights
model.load_state_dict(torch.load("asl_mobilenetv2_best.pth", map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode


# Initialize webcam and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

# Define transforms (EXACTLY as in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match your training size
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# Class labels (A-Z + 0-9)
labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    
]

# Load your trained model
# model = torch.load("asl_mobilenetv2_best.pth", map_location=torch.device('cpu'))
# model.eval()

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        # Get hand bounding box
        x, y, w, h = hands[0]['bbox']
        offset = 20
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:

            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgGray = cv2.equalizeHist(imgGray) 

            # Convert BGR (OpenCV) to RGB (PIL expects this)
            # imgRGB = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            
            # Convert NumPy array to PIL Image
            pil_img = Image.fromarray(imgGray)
            
            # Apply transforms
            img_tensor = transform(pil_img).unsqueeze(0)  # Add batch dimension
            
            # # Predict
            # with torch.no_grad():
            #     output = model(img_tensor)
            #     _, predicted = torch.max(output, 1)
            #     sign = labels[predicted.item()]

            # Predict
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)  # Get probabilities
                confidence, predicted = torch.max(probabilities, 1)
                confidence = confidence.item()  # Convert to Python float
                sign = labels[predicted.item()]

            # Display prediction + confidence
            cv2.putText(
                img, 
                f"Predicted: {sign} ({confidence:.2f})",  # e.g., "A (0.98)"
                (x, y-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

    cv2.imshow("ASL Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#Camera implementation using raspberry pi 5: 
import torch 
from torchvision import transforms, models 
from PIL import Image 
import matplotlib.pyplot as plt 
import cv2 
import os 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# Trained model path 
model_path = "/content/drive/MyDrive/checkpoints/best_model.pth" 
if not os.path.exists(model_path): 
raise FileNotFoundError(f"Model file not found: {model_path}.") 
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Adjust for 4 tumor classes 
model.load_state_dict(torch.load(model_path, map_location=device)) 
model.eval() 
model = model.to(device) 
IMG_SIZE = 224 
transform = transforms.Compose([ 
transforms.Resize((IMG_SIZE, IMG_SIZE)), 
transforms.ToTensor(), 
transforms.Normalize([0.5] * 3, [0.5] * 3) 
]) 
# Class mapping 
class_to_idx = {'glioma': 0, 'meningioma': 1, 'non_tumor': 2, 'pituitary': 3} 
idx_to_class = {v: k for k, v in class_to_idx.items()} 
 
# Function for tumor detection & classification 
def verify_mri_tumor(pil_image, model, transform): 
    image = transform(pil_image).unsqueeze(0).to(device) 
 
    with torch.no_grad(): 
        output = model(image) 
        pred_idx = torch.argmax(output, dim=1).item() 
 
    predicted_class = idx_to_class[pred_idx] 
 
    if predicted_class == "non_tumor": 
        result = "No tumor detected." 
    else: 
        result = f"⚠ Tumor detected: {predicted_class}" 
 
    return result 
 
 
cap = cv2.VideoCapture(0)  # 0 = default camera 
print("Press SPACE to capture an image, ESC to quit.") 
 
captured_image = None 
while True: 
    ret, frame = cap.read() 
    if not ret: 
        print(" Failed to grab frame") 
        break 
 
    cv2.imshow("Camera Feed - Press SPACE to Capture", frame) 
 
    k = cv2.waitKey(1) 
    if k % 256 == 27:  # ESC pressed 
        print(" Exiting without capture...") 
        break 
    elif k % 256 == 32:  # SPACE pressed 
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        captured_image = Image.fromarray(img_rgb)  # Convert to PIL 
        print("Image captured.") 
        break 
 
cap.release() 
cv2.destroyAllWindows() 
 
if captured_image is not None: 
    prediction_result = verify_mri_tumor(captured_image, model, transform) 
    print(prediction_result) 
 
    # Display image with prediction 
    plt.imshow(captured_image) 
    plt.title(prediction_result) 
    plt.axis("off") 
    plt.show()
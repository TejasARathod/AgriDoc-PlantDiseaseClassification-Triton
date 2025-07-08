import os
import torch
import cv2
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Directories
input_dir = "Gradcam_Original"
output_dir = "gradcam_output"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Grad-CAM setup
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Process each image
for file_name in os.listdir(input_dir):
    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(input_dir, file_name)
    print(f"[INFO] Processing {img_path}")

    # Load and preprocess image
    bgr_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    input_tensor = torch.tensor(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to(device)

    # Predict and set target class
    with torch.no_grad():
        outputs = model(input_tensor)
        top_class = outputs.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(top_class)]

    # Generate CAM with smoothing
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
        eigen_smooth=True,
        aug_smooth=True
    )[0, :]  # [0] for first image in batch

    # Visualize and save
    visualization = show_cam_on_image(rgb_img.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
    save_path = os.path.join(output_dir, f"gradcam_{file_name}")
    cv2.imwrite(save_path, visualization[..., ::-1])  # Convert RGB to BGR
    print(f"[SAVED] {save_path}")


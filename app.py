import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
from ultralytics import YOLO  # For YOLOv5

# Load pre-trained models
@st.cache_resource
def load_model(model_name):
    if model_name == "Mask R-CNN (ResNet50)":
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    elif model_name == "Mask R-CNN (ResNet101)":
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, backbone_name='resnet101')
    elif model_name == "Faster R-CNN (ResNet50)":
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif model_name == "YOLOv5":
        model = YOLO("yolov5s.pt")
    elif model_name == "DeepLabV3":
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    else:
        st.error("Invalid model selection!")
        return None
    model.eval()
    return model

# Transform input image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

def segment_with_mask_rcnn(image, model, threshold=0.5):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(image_tensor)

    pred_masks = predictions[0].get('masks', None)  # Masks
    pred_scores = predictions[0]['scores'].numpy()  # Confidence scores
    pred_boxes = predictions[0]['boxes'].numpy()  # Bounding boxes

    segments = []
    if pred_masks is not None:
        pred_masks = pred_masks.squeeze(1).numpy()  # Masks
        for i, score in enumerate(pred_scores):
            if score > threshold:
                mask = (pred_masks[i] > 0.5).astype(np.uint8)  # Binarize mask
                bbox = pred_boxes[i].astype(int)

                # Apply mask to isolate object
                segmented_image = np.array(image) * mask[:, :, None]

                # Crop object from bounding box
                cropped_object = segmented_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                segments.append(cropped_object)
    else:
        for i, score in enumerate(pred_scores):
            if score > threshold:
                bbox = pred_boxes[i].astype(int)
                cropped_object = np.array(image)[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                segments.append(cropped_object)
    return segments

def segment_with_yolo(image, model, threshold=0.5):
    image_np = np.array(image)
    results = model.predict(source=image_np, conf=threshold, save=False)
    segments = []
    for result in results[0].boxes:
        bbox = result.xyxy[0].cpu().numpy().astype(int)  # Bounding box
        cropped_object = image_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        segments.append(cropped_object)
    return segments

def segment_with_deeplabv3(image, model, threshold=0.5):
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    masks = (output.argmax(0).byte().cpu().numpy() > 0).astype(np.uint8)
    segments = []
    contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped_object = np.array(image)[y:y+h, x:x+w]
        segments.append(cropped_object)
    return segments

# Streamlit UI
def main():
    st.title("Image Segmentation App")
    st.write("Upload a raster image, select a model, and extract segments.")

    # Model selection dropdown
    model_name = st.selectbox(
        "Choose a segmentation model:",
        ["Mask R-CNN (ResNet50)", "Mask R-CNN (ResNet101)", "Faster R-CNN (ResNet50)", "YOLOv5", "DeepLabV3"]
    )

    # Confidence threshold slider
    threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    # Upload Image
    uploaded_file = st.file_uploader("Choose a raster image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load model
        model = load_model(model_name)

        if model is not None:
            # Segment and display results
            with st.spinner("Processing..."):
                if "Mask R-CNN" in model_name or "Faster R-CNN" in model_name:
                    segments = segment_with_mask_rcnn(image, model, threshold)
                elif model_name == "YOLOv5":
                    segments = segment_with_yolo(image, model, threshold)
                elif model_name == "DeepLabV3":
                    segments = segment_with_deeplabv3(image, model, threshold)
                else:
                    st.error("Model not supported!")
                    return

                st.success(f"Found {len(segments)} segments!")

            # Display and save segments
            for i, segment in enumerate(segments):
                st.image(segment, caption=f"Segment {i+1}", use_column_width=True)
                segment_image = Image.fromarray(segment)
                segment_image.save(f"segment_{i+1}.png")
                st.write(f"Segment {i+1} saved as segment_{i+1}.png")

if __name__ == "__main__":
    main()

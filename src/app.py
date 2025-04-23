import streamlit as st
import matplotlib.pyplot as plt
from model import UNet
import torch
import nibabel as nib
import numpy as np
import io
import tempfile
import shutil
import os

# Set up Streamlit title and description
st.title("MedSeg Body Part Detection Viewer")

# Load the pre-trained model
model = UNet()
model.load_state_dict(torch.load(r'D:/VLM/models/unet_medseg.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Upload the file and read the .nii.gz image
uploaded_file = st.file_uploader("Upload a .nii.gz image", type=["nii.gz", ".gz"])

if uploaded_file:
    # Create a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    # Load the NIfTI image using nibabel from the temporary file path
    img = nib.load(temp_file_path).get_fdata()
    
    # Extract a slice from the image (here, we're using the middle slice)
    slice_img = torch.tensor(img[:, :, img.shape[2] // 2], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Run the model to predict the segmentation mask
    with torch.no_grad():
        pred = model(slice_img).squeeze().numpy()

    # Display the original and predicted images using Matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(slice_img.squeeze(), cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(slice_img.squeeze(), cmap='gray')
    ax[1].imshow(pred, cmap='jet', alpha=0.5)
    ax[1].set_title("Predicted Mask")
    
    # Display the plot in the Streamlit app
    st.pyplot(fig)
    
    # Clean up the temporary file after use
    os.remove(temp_file_path)  # Remove the temporary file correctly
    # Remove the temporary file

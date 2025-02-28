3D Segmentation Model for Abdominal Organs
Overview
This project aims to build a 3D segmentation model to identify and segment abdominal organs (Liver, Right Kidney, Left Kidney, and Spleen) from CT scan images. The model is built using a modified version of the VNet architecture, which is specifically designed for 3D medical image segmentation tasks.

The primary goal of this project is to achieve accurate segmentation of the specified organs, which can be used for various applications in medical imaging and diagnostics.

Dataset
The dataset used in this project is the CT Abdomen Organ Segmentation Dataset, which contains 3D CT scans and corresponding segmentation masks for different abdominal organs. The dataset is preprocessed by normalizing the scans and resizing them to a consistent size.

Model Architecture
The model is based on the VNet architecture, a popular choice for 3D medical image segmentation. The VNet model is designed to learn hierarchical features through a series of encoder and decoder layers. The encoder downsamples the input image to capture high-level features, while the decoder upsamples the encoded features to produce a segmentation map of the same size as the input image.

Key Features:
3D Convolutional Layers: For capturing spatial relationships in 3D images.
Batch Normalization: To stabilize and accelerate the training process.
ReLU Activation: For non-linear transformations.
3D Max Pooling: For downsampling in the encoder.
3D Transposed Convolutions: For upsampling in the decoder.
Setup Instructions
1. Clone the Repository
git clone https://github.com/yourusername/3d-segmentation-model.git
cd 3d-segmentation-model

### **2. Install Dependencies**
Ensure that you have Python 3.7+ installed. Install the required Python packages using the `requirements.txt` file:

``
pip install -r requirements.txt

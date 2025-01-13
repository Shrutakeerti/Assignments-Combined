#  Programming Test: Learning Activations in Neural Networks

## Overview
This project is a Breast Cancer Prediction System that utilizes a Neural Network model to predict whether a tumor is malignant or benign based on input features. Additionally, the project integrates a chatbot assistant to guide users in using the system or answer basic questions.

## Features
- **Breast Cancer Prediction**: Enter tumor features to get a prediction (malignant or benign).
- **Interactive Chatbot**: A simple rule-based chatbot to assist users with the prediction process.
- **Streamlit UI**: User-friendly interface built using Streamlit for both prediction and chatbot interactions.
- **Flask API**: Backend server powered by Flask that handles prediction requests and chatbot interactions.

## Table of Contents
- Project Structure
- Installation
- Usage
- Endpoints
- Technology Stack
- Screenshots
- License
- Contributors
- Contact

## Project Structure
```bash
.
├── app.py                 # Flask API for prediction and chatbot
├── streamlit_app.py        # Streamlit app for prediction UI and chatbot interface
├── model.pth               # Pretrained PyTorch model
├── README.md               # Project readme file
└── requirements.txt        # List of dependencies
```
## Installation

### 1.Clone the repository:
```bash
git clone https://github.com/your-username/breast-cancer-prediction-chatbot.git
cd breast-cancer-prediction-chatbot
```
### 2. Set up a virtual environment:

#### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
#### For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
#### Download the pretrained model:
#### Make sure you have the model.pth file in the project directory.
## Usage

### Start the Flask API:
```bash
python app.py
```
### This will start the Flask server on http://127.0.0.1:5000. The server handles both prediction and chatbot routes.

### Run the Streamlit App.In a separate terminal, run the following command to launch the Streamlit app:
```bash
streamlit run streamlit_app.py

```

## Endpoints

### Prediction Endpoint:
- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:
```json
{
  "features": [feature_1, feature_2, ..., feature_31]
}
```

- **Response**:
```json
{
  "prediction": "malignant" or "benign"
}
```
### Chatbot Endpoint:
- **URL**: `/chatbot`
- **Method**: `POST`
- **Request Body**:
```json
{
  "message": "user's message"
}
```
- **Response**:
```json
{
  "response": "chatbot's response"
}
```
## Technology Stack
- **Frontend**: Streamlit (for UI)
- **Backend**: Flask (API)
- **ML Model**: PyTorch
- **Libraries**:
  - `torch`: For neural network model
  - `pandas`, `numpy`: Data manipulation
  - `sklearn`: Data preprocessing (StandardScaler)
  - `requests`: API requests

## Screenshot

![Chatbot](https://github.com/Shrutakeerti/Assignment---27-09-2024/blob/main/Chatbot%20(2).png)
![Prediction features](https://github.com/Shrutakeerti/Assignment---27-09-2024/blob/main/prediction.jpeg)




## This is deployed using streamlit

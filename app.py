# app.py

import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Define the AdaAct activation function
class AdaAct(nn.Module):
    def __init__(self):
        super(AdaAct, self).__init__()
        self.k0 = nn.Parameter(torch.randn(1))
        self.k1 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.k0 + self.k1 * x

# Define the Neural Network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ada_act1 = AdaAct()  # Custom activation
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ada_act1(out)
        out = self.fc2(out)
        return out

# Load the trained model
input_size = 31  # Use the same size as during training, e.g., 31 if that was used
hidden_size = 30
output_size = 2
model = NeuralNet(input_size, hidden_size, output_size)

# Correct the path with a raw string
model.load_state_dict(torch.load(r'D:\GDG\model (2).pth')) 
model.eval()  # Set the model to evaluation mode


# Define the root route for GET request
@app.route('/', methods=['GET'])
def home():
    return "API is running. Use POST /predict to make predictions."




# Define the route for the API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting a JSON request body with feature values
    if not data or 'features' not in data:
        return jsonify({"error": "Please provide input features in the 'features' field."}), 400

    # Extract features from the request and convert them to a numpy array
    features = np.array(data['features'])

    # Ensure input shape is correct
    if features.shape[0] != input_size:
        return jsonify({"error": f"Input features must be of length {input_size}."}), 400

    # Standardize the input features
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(1, -1))

    # Convert features to PyTorch tensor
    features_tensor = torch.FloatTensor(features)

    # Make prediction
    with torch.no_grad():
        output = model(features_tensor)
        _, predicted = torch.max(output.data, 1)
    
    diagnosis = 'malignant' if predicted.item() == 1 else 'benign'
    
    return jsonify({"prediction": diagnosis})


# Simple chatbot route
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Please provide a message."}), 400
    
    user_message = data['message'].lower()
    
    # Simple chatbot logic (rule-based)
    if "hello" in user_message:
        response = "Hello! How can I help you with breast cancer prediction?"
    elif "predict" in user_message:
        response = "You can use the /predict endpoint to get a breast cancer prediction."
    else:
        response = "Sorry, I didn't understand that. Try asking about predictions or say hello!"
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

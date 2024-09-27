# breast_cancer_app.py

import streamlit as st
import requests
import numpy as np

# Set the title of the app
st.title("Breast Cancer Prediction System")

# List of meaningful feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'points_mean', 'symmetry_mean', 'dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'points_se', 'symmetry_se', 'dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'points_worst', 'symmetry_worst', 'dimension_worst'
]

# Input fields for the features with their names
features = []
for feature_name in feature_names:
    feature_value = st.number_input(f"{feature_name}", value=0.0)  # Set default value as 0.0
    features.append(feature_value)

# Button to make prediction
if st.button("Predict"):
    # Prepare the data to be sent to the Flask API
    input_data = {"features": features}
    
    # Send a POST request to the Flask API
    response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
    
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")


# streamlit_app.py

import streamlit as st
import requests

# Set the title of the app
st.title("Breast Cancer Prediction System with Chatbot")

# Tab structure: Chatbot and Prediction
tab1, tab2 = st.tabs(["Chatbot", "Prediction"])

# Chatbot Tab
with tab1:
    st.header("Chat with the Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for msg in st.session_state.messages:
        if msg["is_user"]:
            st.text_area("You: ", value=msg["text"], height=50, key=f"user_{msg['id']}", disabled=True)
        else:
            st.text_area("Bot: ", value=msg["text"], height=50, key=f"bot_{msg['id']}", disabled=True)

    # User input
    user_input = st.text_input("Type a message:")
    
    if st.button("Send"):
        if user_input:
            # Save user message to history
            st.session_state.messages.append({"is_user": True, "text": user_input, "id": len(st.session_state.messages)})

            # Send request to chatbot API
            chatbot_response = requests.post("http://127.0.0.1:5000/chatbot", json={"message": user_input}).json()

            # Save bot response to history
            st.session_state.messages.append({"is_user": False, "text": chatbot_response["response"], "id": len(st.session_state.messages)})


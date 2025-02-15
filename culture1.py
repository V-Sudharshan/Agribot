import streamlit as st
import google.generativeai as genai
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import pyttsx3
import threading
import random
import speech_recognition as sr
import os

# Configure Google Gemini AI API Key
genai.configure(api_key="AIzaSyCHx34_xg5jpTJlNS8b2FX8iSvsK_Qo3n8")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")

# Load Pre-trained Model for Image Analysis (MobileNetV2)
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()

# Add custom CSS for styling
def add_custom_css():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <style>
        body {
            font-family: 'Roboto', sans-serif;
        }
        .stApp {
            background-color: #f0f0f5;  /* Light background color */
            color: #333;  /* Dark text color */
        }
        h1 {
            color: #2c3e50;  /* Darker color for headers */
            font-size: 36px;  /* Title font size */
            text-align: center;  /* Center align title */
        }
        h2 {
            color: #2c3e50;  /* Darker color for headers */
            text-align: center;  /* Center align subtitle */
            font-size: 24px;  /* Subtitle font size */
        }
        .stTextInput {
            font-size: 18px;  /* Increase font size for input */
            padding: 10px;  /* Add padding */
            border-radius: 5px;  /* Rounded corners */
            border: 1px solid #ccc;  /* Light border */
            width: 80%;  /* Set width of the input */
            margin: 0 auto;  /* Center the input */
            display: block;  /* Make it a block element */
        }
        .stButton {
            background-color: transparent;  /* Transparent button color */
            color: #007bff;  /* Blue text */
            border-radius: 5px;  /* Rounded corners */
            padding: 10px 20px;  /* Padding */
            margin-top: 10px;  /* Space above the button */
            float: right;  /* Align button to the right */
            border: 1px solid #007bff;  /* Blue border */
        }
        .stButton:hover {
            background-color: #e0e0e0;  /* Light gray on hover */
        }
        .stMarkdown {
            font-size: 16px;  /* Increase font size for markdown */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to add custom CSS
add_custom_css()

# Set Background Image
def set_background(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
        encoded = base64.b64encode(img_data).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}"); 
            background-size: cover;
            background-position: center;
            color: white;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# Set an agriculture-themed background
set_background("img_3.png")  # Replace with your image path

# Text-to-Speech Function
def speak(text):
    def speak_thread():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=speak_thread).start()

# Ask Agriculture Question (Google Gemini AI)
def ask_agriculture_question(question):
    try:
        response = model.generate_content([question], request_options={"timeout": 10})
        if response and hasattr(response, 'text'):
            answer = response.text.strip()
            st.success(f"🤖 AI Response: {answer}")
            speak(answer)

            # Recommend seeds based on the question
            recommend_seeds(question)
        else:
            st.error("⚠ No response received.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Recommend seeds based on the question
def recommend_seeds(question):
    seeds = {
        "cotton": {"name": "Cotton Seed", "price": "₹500"},
        "tomato": {"name": "Tomato Seed", "price": "₹200"},
        "wheat": {"name": "Wheat Seed", "price": "₹400"},
        "rice": {"name": "Rice Seed", "price": "₹450"},
        "corn": {"name": "Corn Seed", "price": "₹300"},
        "sunflower": {"name": "Sunflower Seed", "price": "₹350"},
    }

    recommended = []
    for key in seeds.keys():
        if key in question.lower():
            recommended.append(seeds[key])

    if recommended:
        st.subheader("Recommended Seeds:")
        for seed in recommended:
            st.write(f"{seed['name']} - {seed['price']}")

# Detect Crop Disease
def detect_crop_disease(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = mobilenet_model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    top_prediction = decoded_predictions[0]
    class_name, class_desc, class_prob = top_prediction

    result = f"Detected Crop Disease: {class_desc} ({class_prob * 100:.2f}%)"
    st.success(result)
    speak(result)

    solution = get_solution_for_disease(class_desc)
    st.subheader("Solution")
    st.write(solution)
    speak(solution)

# Solution based on Disease Type
def get_solution_for_disease(disease_desc):
    if "fungus" in disease_desc.lower():
        return "Use a fungicide like Mancozeb."
    elif "bacteria" in disease_desc.lower():
        return "Apply copper-based bactericides."
    else:
        return "Consult an expert."

# Analyze Soil Health Image
def analyze_soil_health_image(image):
    soil_quality = "Good"
    soil_ph = np.random.uniform(5.5, 7.5)
    soil_moisture = np.random.uniform(15, 40)

    st.subheader("Soil Health Analysis")
    st.write(f"*Soil Quality:* {soil_quality}")
    st.write(f"*Soil pH:* {soil_ph:.2f}")
    st.write(f"*Soil Moisture Content:* {soil_moisture:.2f}%")

    solution = "Maintain regular composting."
    st.subheader("Solution")
    st.write(solution)
    speak(solution)

# Predict Market Price for a Crop
def predict_market_price(crop_name):
    market_prices = {"wheat": 2500, "rice": 3200, "corn": 1500}
    price = market_prices.get(crop_name.lower(), random.randint(1000, 5000))
    st.subheader(f"Market Price for {crop_name.capitalize()}")
    st.write(f"Estimated Price: ₹{price} per quintal")
    speak(f"Estimated market price for {crop_name} is ₹{price} per quintal.")

# Detect Insects in Crop Image
def detect_insects(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = mobilenet_model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    insect_name = decoded_predictions[0][1]

    st.success(f"Detected Insect: {insect_name}")
    speak(f"Detected insect: {insect_name}")

    solution = get_solution_for_insect(insect_name)
    st.subheader("Solution")
    st.write(solution)
    speak(solution)

# Solution based on Insect Type
def get_solution_for_insect(insect_name):
    if "aphid" in insect_name.lower():
        return "Use neem oil or insecticidal soap."
    elif "caterpillar" in insect_name.lower():
        return "Apply Bacillus thuringiensis (Bt) spray."
    else:
        return "Consult an expert."

# Fertilizer Recommendation for Crop
def recommend_fertilizer(crop_name):
    fertilizers = {
        "wheat": "Use NPK 20:20:20 for optimal growth.",
        "rice": "Apply Urea and DAP in proper proportions.",
        "corn": "Use Potassium-rich fertilizers."
    }

    recommendation = fertilizers.get(crop_name.lower(), "Consult an expert for best fertilizer recommendations.")
    st.subheader(f"Fertilizer Recommendation for {crop_name.capitalize()}")
    st.write(recommendation)
    speak(recommendation)

# Crop Rotation Suggestions
def crop_rotation_suggestions(crop_name):
    rotation_suggestions = {
        "wheat": "Rotate with legumes like peas or beans for better soil health.",
        "rice": "Rotate with vegetables like tomato or garlic to break pest cycles.",
        "corn": "Rotate with soybeans or cover crops like clover."
    }

    suggestion = rotation_suggestions.get(crop_name.lower(), "Consult an expert for crop rotation advice.")
    st.subheader(f"Crop Rotation Suggestions for {crop_name.capitalize()}")
    st.write(suggestion)
    speak(suggestion)

# Crop Equipment Suggestions
def crop_equipment_suggestions(crop_name):
    equipment_suggestions = {
        "wheat": "Use a seed drill for planting and a combine harvester for harvesting.",
        "rice": "A rice transplanter and a harvester are essential for efficient farming.",
        "corn": "A corn planter and a combine harvester are recommended for large-scale farming."
    }

    suggestion = equipment_suggestions.get(crop_name.lower(), "Consult an expert for appropriate equipment.")
    st.subheader(f"Equipment Suggestions for {crop_name.capitalize()}")
    st.write(suggestion)
    speak(suggestion)

# Function to listen to the farmer's voice input
def listen_to_farmer():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening for your question...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        audio = recognizer.listen(source)  # Listen for the audio input
        st.write("🎤 You spoke!")

    try:
        # Recognizing the speech using Google Web Speech API
        query = recognizer.recognize_google(audio)
        st.write(f"Farmer said: {query}")
        return query
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand the audio.")
        return None
    except sr.RequestError:
        st.write("Sorry, there was an error with the speech recognition service.")
        return None

# Main Streamlit App Layout
st.title("🌾 AgriBot - Your AI Agriculture Assistant")
st.header("Welcome to AgriBot - A Smart Assistant for Farmers 🌾")

# Enhance Sidebar with icons and improved navigation
option = st.sidebar.radio("Select an option:", [
    "Ask AgriBot", "Detect Crop Disease", "Analyze Soil Health",
    "Predict Market Price", "Detect Insects", "Fertilizer Recommendation",
    "Crop Rotation Suggestions", "Crop Equipment Suggestions", "Voice Interaction"
])

if option == "Ask AgriBot":
    user_query = st.text_input("Ask any agriculture-related question:", placeholder="Type your question here")
    if st.button("Ask"):
        ask_agriculture_question(user_query)
elif option == "Detect Crop Disease":
    uploaded_image = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        detect_crop_disease(image)
elif option == "Analyze Soil Health":
    uploaded_image = st.file_uploader("Upload a soil image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        analyze_soil_health_image(image)
elif option == "Predict Market Price":
    crop_name = st.text_input("Enter crop name:")
    if st.button("Predict"):
        predict_market_price(crop_name)
elif option == "Detect Insects":
    uploaded_image = st.file_uploader("Upload an insect image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        detect_insects(image)
elif option == "Fertilizer Recommendation":
    crop_name = st.text_input("Enter crop name:")
    if st.button("Recommend Fertilizer"):
        recommend_fertilizer(crop_name)
elif option == "Crop Rotation Suggestions":
    crop_name = st.text_input("Enter crop name:")
    if st.button("Suggest Rotation"):
        crop_rotation_suggestions(crop_name)
elif option == "Crop Equipment Suggestions":
    crop_name = st.text_input("Enter crop name:")
    if st.button("Suggest Equipment"):
        crop_equipment_suggestions(crop_name)
elif option == "Voice Interaction":
    if st.button("Start Listening"):
        question = listen_to_farmer()
        if question:
            ask_agriculture_question(question)

# Display available seeds and fertilizers
st.markdown("## 🌿 Available Seeds and Fertilizers")

# Get the absolute path of the images directory (without trailing slash)
IMAGE_DIR = os.path.abspath("pics")

# Define product details with image paths
products = [
    {"name": "Cotton Seed", "price": "₹500", "image": "cottonn.jpg"},
    {"name": "Tomato Seed", "price": "₹200", "image": "tomato.jpg"},
    {"name": "Wheat Seed", "price": "₹400", "image": "wheat_seed.jpg"},
    {"name": "Rice Seed", "price": "₹450", "image": "rice_seed.jpg"},
    {"name": "Corn Seed", "price": "₹300", "image": "corn_seed.jpg"},
    {"name": "Sunflower Seed", "price": "₹350", "image": "sunflower_seed.jpg"},
    {"name": "Grommar Fertilizer", "price": "₹300", "image": "gromor_fertilizer.jpg"},
    {"name": "NDP Fertilizer", "price": "₹350", "image": "NDP_Fertilizer.jpg"},
    {"name": "Urea Fertilizer", "price": "₹250", "image": "urea_Fertilizer.jpg"},
    {"name": "DAP Fertilizer", "price": "₹400", "image": "dpa_fertilizer.jpg"},
    {"name": "Potash Fertilizer", "price": "₹380", "image": "Potash_Fertilizer.jpg"},
    {"name": "Organic Compost", "price": "₹280", "image": "organic_Compost.jpg"},
]

# Display products in a horizontal layout with images
cols = st.columns(4)  # Adjust columns for layout

for i, product in enumerate(products):
    with cols[i % 4]:  # Wrap to the next row after 4 items
        image_path = os.path.join(IMAGE_DIR, product["image"])  # Use absolute path

        if os.path.exists(image_path):  # Check if image exists
            st.image(image_path, width=100)  # Display small image
        else:
            st.warning(f"⚠ Image not found: {image_path}")  # Show full path for debugging

        st.write(f"{product['name']}")
        st.write(f"💰 {product['price']}")

# Final touches
st.markdown("### Thank you for using AgriBot! 🌱")
import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

# Set up the app title and layout
st.title("👤 Gender Classification using ViT")
st.write("Upload an image to predict the gender of the person.")

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_classifier():
    return pipeline(
        "image-classification",
        model="rizvandwiki/gender-classification"
    )

gender_classifier = load_classifier()

# File uploader for user images
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        # Classify gender
        gender_predictions = gender_classifier(image)

        # Sort predictions by score, highest first
        gender_predictions = sorted(
            gender_predictions,
            key=lambda x: x["score"],
            reverse=True
        )

        # Display top result
        top_prediction = gender_predictions[0]

        st.success(f"**Predicted Gender: {top_prediction['label']}**")
        st.write(f"Confidence Score: {top_prediction['score']:.2%}")

        # Show all probabilities in a chart
        with st.expander("See detailed probabilities"):
            chart_data = pd.DataFrame({
                "Gender": [p["label"] for p in gender_predictions],
                "Confidence": [p["score"] for p in gender_predictions]
            }).set_index("Gender")

            st.bar_chart(chart_data)

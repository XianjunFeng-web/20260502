
import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Age Classification", page_icon="👤")
st.title("Age Classification using ViT")
st.write("Upload an image of a face to predict the age range.")

# 2. Load the model (cached to prevent reloading on every interaction)
@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

age_classifier = load_classifier()

# 3. Sidebar / File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a PIL Image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Classifying..."):
            # 4. Classify age
            age_predictions = age_classifier(image)
            # Sort by highest score (confidence)
            age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
            
            # 5. Display results
            top_prediction = age_predictions[0]
            st.success(f"**Predicted Age Range:** {top_prediction['label']}")
            st.metric("Confidence Score", f"{top_prediction['score']:.2%}")
            
            # Show additional details in an expander
            with st.expander("View all predictions"):
                for pred in age_predictions:
                    st.write(f"- {pred['label']}: {pred['score']:.4f}")
else:
    st.info("Please upload an image to start classification.")

import streamlit as st
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import base64

# Set page config
st.set_page_config(layout="wide")

# Function to get image as base64
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Get image as base64
img = get_img_as_base64("image.jpg")

# Custom CSS for setting background image and sidebar animation
page_bg_img = f"""
<style>
body {{
    background-color: #f0f2f6;
}}
[data-testid="stAppSidebar"] > div[role="navigation"] {{
    transition: all 0.3s;
    position: fixed !important;
    width: 250px !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
    z-index: 1 !important;
}}
[data-testid="stAppSidebar"][aria-expanded="false"] > div[role="navigation"] {{
    width: 0 !important;
    overflow: hidden !important;
}}
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://blog.dataiku.com/hubfs/towfiqu-barbhuiya-oZuBNC-6E2s-unsplash-1.jpg#keepProtocol");
    background-size: cover;
    background-position: center;
}}
.stButton>button {{
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    cursor: pointer;
    border-radius: 5px;
    transition: background-color 0.3s ease;
}}
.stButton>button:hover {{
    background-color: #45a049;
    box-shadow: 0px 0px 10px #888888;
}}
.stTextInput>div>div>input {{
    border-radius: 5px;
}}
.stTitle {{
    font-family: "Arial Black", Arial, sans-serif;
    font-size: 36px;
    color: #333333;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}}
.stHeader {{
    font-family: "Arial", sans-serif;
    font-size: 22px;
    color: #666666;
}}
.stText {{
    font-family: "Arial", sans-serif;
    font-size: 18px;
    color: #000000;
}}
.stButton:hover {{
    box-shadow: 0px 0px 10px #888888;
}}
.st-emotion-cache-nahz7x e1nzilvr5
{{
    box-shadow: 0px 0px 10px #888888;
}}
</style>
"""

# Apply custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

# Initialize ViltProcessor and ViltForQuestionAnswering
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Function to capitalize the first letter of each word
def capitalize_first_letter(text):
    return text.title()

# Set up the Streamlit app
st.title("Visual Question Answering")
st.header("Upload an image and enter a question to get an answer.")

# Sidebar
st.sidebar.title("Menu")
menu_items = {
    "Home": ":house:",
    "About Us": ":information_source:",
    "Contact Us": ":email:"
}
selected_menu_item = st.sidebar.radio("", list(menu_items.keys()))

# Create columns for image upload, question input, and sidebar
col1, col2, col3 = st.columns([4, 4, 1])

# Image upload
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

# Question input and clear button
with col2:
    question = st.text_input("Enter your question here:")
    question = capitalize_first_letter(question)
    if st.button("Clear"):
        question = ""

    # Process the image and question when both are provided
    if uploaded_file and question:
        st.write("### Your Question:")
        st.write(f"*{question}*")

        # Function to get the answer
        def get_answer(image, text):
            try:
                # Process the image
                encoding = processor(image, text, return_tensors="pt")
                outputs = model(**encoding)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                answer = model.config.id2label[idx]
                return answer
            except Exception as e:
                return str(e)

        # Get the answer
        with st.spinner('Searching for answer...'):
            answer = get_answer(image, question)

        # Capitalize the first letter of each word in the answer
        answer = capitalize_first_letter(answer)

        # Display the answer below the question input
        st.write("### Answer:")
        st.success(f"*{answer}*")
    elif question:  # If question is provided but no image
        st.warning("Image is not found")  # Display warning

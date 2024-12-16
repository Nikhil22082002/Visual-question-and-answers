Visual Question Answering Web App

This project is a Visual Question Answering (VQA) Web App built using Streamlit. It leverages pre-trained Transformer models to process images and answer questions related to them.

Features

Upload images and ask questions about their content.

Uses a Transformer-based model for question answering.

Simple and intuitive user interface built with Streamlit.

Libraries Used

Streamlit: For creating the web app interface.

PIL (Python Imaging Library): For image processing.

Transformers: Specifically, ViltProcessor and ViltForQuestionAnswering from the Transformers library by Hugging Face for handling visual question answering.

Base64: For encoding and decoding images when necessary.

Additional Models

This project has also experimented with ResNet for feature extraction, but the ResNet model implementation is not included in this repository due to external dependencies.

Installation

Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

Steps

Clone the repository:

git clone https://github.com/your-repository-name.git
cd your-repository-name

Install the required Python packages:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

How to Use

Launch the app as described in the installation steps.

Upload an image using the file uploader.

Enter your question about the uploaded image.

The app will process the input and provide an answer.

Code Overview

Main Libraries

Streamlit: Handles the app's interface.

PIL (Image): Opens and preprocesses the uploaded image.

Transformers (ViltProcessor, ViltForQuestionAnswering): Processes the image and question to produce an answer.

Base64: Supports encoding for image-related operations.

Core Workflow

Users upload an image and input a question.

The image is preprocessed using PIL.

The text and image inputs are passed through the ViltProcessor.

The ViltForQuestionAnswering model processes the inputs and generates a response.

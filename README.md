# Dental Implants Image Classification Model - UChicago Applied Data Science Master Capstone Project

This repository contains the code and resources for the Capstone project, which focuses on identifying the model of dental implants from X-ray images using machine learning models.

## Repository Structure

- **`/data/`**: 
  - Contains the original data used for training the models.
  - Note: Additional directories created during training (e.g., temporary files or preprocessed data) are not included here.

- **`/model/`**: 
  - Contains Jupyter notebooks for different models.
  - **Note**: The Siamese model must be executed in Google Colab to take advantage of their GPU runtime.

- **`/app/`**:
  - Contains a demo of the application built using Streamlit.
  - To run the app, navigate to the `app` folder in your terminal and use the following command:
    ```bash
    streamlit run app.py
    ```
  - **Missing Files**: The `./app/data/` folder is missing `.pt` files needed to load the model directly, as these files are too large for GitHub. To generate these files, train the models yourself using the provided scripts.

## Project Overview

This project provides a two-step process to identify the model of dental implants from X-ray images:

1. **Implant Localization**:
   - Uses a YOLO model to detect the locations of implants within the X-ray image.
   
2. **Model Identification**:
   - After localization, another model identifies the specific type of implant present in each detected location.

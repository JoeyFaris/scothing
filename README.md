# Clothing Classification Application

This project is a web application that takes an image of clothing, analyzes it using a machine learning model, and provides feedback on the type of clothing (e.g., shirt, pants, hat). The app uses a deep learning model fine-tuned on the **DeepFashion** dataset to classify clothing items.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model](#model)
- [License](#license)

## Overview

The application consists of:
- **Backend API**: Accepts images via an API call, processes them using a pre-trained model fine-tuned on the DeepFashion dataset, and returns the predicted clothing category.
- **Frontend**: A simple interface where users can upload an image and see the clothing type predicted by the model.

## Features
- Upload an image and get a prediction of clothing type (shirt, pants, hat, etc.)
- Pre-trained model fine-tuned on the DeepFashion dataset for robust predictions
- Simple, user-friendly web interface

## Requirements

The project requires the following dependencies:

- Python 3.8+
- TensorFlow or PyTorch (depending on your chosen deep learning framework)
- Flask or FastAPI for the API backend
- React (or any other JS framework) for the frontend
- DeepFashion dataset (download [here](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html))

### Python Libraries
- `tensorflow` or `torch`
- `flask` or `fastapi`
- `pillow`
- `numpy`
- `opencv-python`
- `scikit-learn`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/clothing-classification-app.git
cd clothing-classification-app
```

2. Set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the DeepFashion dataset and place it in the `dataset/` folder

## Usage

### 1. Train the Model
To train the model on the DeepFashion dataset:
```bash
python model/train.py
```

### 2. Start the Backend API
To start the API for image classification:
```bash
cd api/
python app.py
```
The API will run on http://localhost:5000

### 3. Start the Frontend
Move to the frontend/ folder, install dependencies, and start the frontend server:
```bash
cd frontend/
npm install
npm start
```
The frontend will be accessible at http://localhost:3000

## File Structure
```
clothing-classification-app/
│
├── dataset/                        # DeepFashion dataset
│   ├── images/                     # Image files
│   ├── labels.csv                  # Labels mapping to clothing categories
│
├── model/                          # Model-related files
│   ├── train.py                    # Script for training the model
│   ├── model.py                    # Model architecture (e.g., ResNet)
│   ├── evaluate.py                 # Script to evaluate model performance
│   ├── saved_model/                # Saved model checkpoints
│
├── api/                            # Backend API for image classification
│   ├── app.py                      # Flask/FastAPI app for inference
│   ├── utils.py                    # Helper functions (image preprocessing, prediction)
│
├── frontend/                       # Frontend files (React app)
│   ├── public/                     # Public assets (e.g., index.html)
│   ├── src/                        # React source files
│   ├── App.js                      # Frontend logic for image upload and display
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Ignoring unnecessary files
```

## Model
The model used in this project is based on a pre-trained deep learning architecture (e.g., ResNet, VGG, MobileNet) fine-tuned on the DeepFashion dataset. The model predicts clothing categories such as "shirt", "pants", "hat", etc.

If you want to customize the model, you can modify `model/model.py` and re-train it using `model/train.py`.

## License
This project is licensed under the MIT License.

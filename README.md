# Clothing Classification Application

This is a machine learning-powered web application that classifies clothing items from user-uploaded photos. The application includes a frontend for user interaction, a backend API for image processing and classification, and a machine learning model for analyzing clothing types. The project is containerized using Docker for easy deployment.

## Features

- **Machine Learning Model**: A Convolutional Neural Network (CNN) trained to classify clothing items.
- **REST API**: Built with Flask to handle image uploads and return predictions.
- **Frontend**: A React-based UI for users to upload photos and view classification results.
- **Dockerized Deployment**: Uses Docker and Docker Compose for multi-service orchestration.
- **Automated Training**: Scripts for running model training in a containerized environment.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)

## Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Node.js (for frontend)
- Python 3.x (for API and machine learning)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/joeyfaris/scothing.git
cd scothing
```

2. Build and run the containers:
```bash
docker-compose up --build
```

The application will be accessible at http://localhost:3000 for the frontend and http://localhost:5000 for the API.

## Usage

### Frontend
- Access the frontend at http://localhost:3000
- Upload an image of clothing
- The API will analyze the image and return the predicted clothing type

### API
Send a POST request to the API at http://localhost:5000/predict with an image file.
The API will return a JSON response with the predicted clothing type.

Example using curl:
```bash
curl -X POST -F "file=@path/to/image.jpg" http://localhost:5000/predict
```

### Model Training
To train the model, use the provided script:
```bash
./scripts/run_training.sh
```
This will start the training process using the data in the dataset/ folder.

## Project Structure

```plaintext
├── README.md               # Project documentation
├── api                     # Backend API
│   ├── Dockerfile           # Dockerfile for API service
│   ├── app.py               # Main API logic (Flask)
│   ├── requirements.txt     # API dependencies
│   └── utils.py             # Helper functions for API
├── dataset                  # Dataset folder
│   └── labels.csv           # Image labels for model training
├── docker-compose.yml       # Docker Compose configuration
├── frontend                 # Frontend UI (React)
│   ├── App.js               # Main frontend component
│   └── package.json         # Frontend dependencies
├── main.py                  # Entry point for the application
├── model                    # Machine learning model code
│   ├── config.py            # Configuration settings for the model
│   ├── evaluate.py          # Model evaluation script
│   ├── model.py             # Model architecture
│   └── train.py             # Model training logic
├── requirements.txt         # Root Python dependencies
├── scripts                  # Shell scripts for automation
│   ├── run_training.sh      # Script to run model training
│   └── setup.sh             # Setup script
└── tests                    # Unit and integration tests
    ├── test_api.py          # Tests for API
    └── test_model.py        # Tests for machine learning model
```


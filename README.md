# Shoplifting Detection using CNLSTM2D Model

## Overview

This repository contains a machine learning model for detecting shoplifting activity in CCTV footage using a Convolutional Neural Network with Long Short-Term Memory in 2D (CNLSTM2D). The model is built using TensorFlow and Keras and requires the installation of the following dependencies: TensorFlow, Keras, OpenCV, and NumPy.

## Getting Started

Follow the steps below to get started with the shoplifting detection model:

### Requirements

- TensorFlow
- Keras
- OpenCV
- NumPy

You can install the required packages using the following command:

```bash
pip install tensorflow keras opencv-python numpy
```

### Setting up

1. Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/shoplifting-detection.git
cd shoplifting-detection
```

2. Open the `model_testing.py` file and update the `input_video_path` variable with the path to the CCTV footage you want to test.

3. Save the changes.

### Running the Model

Execute the following command to run the model testing script:

```bash
python model_testing.py
```

The script will load the pre-trained CNLSTM2D model and apply it to the specified CCTV footage to detect shoplifting activity.

## Model Details

The shoplifting detection model is based on a CNLSTM2D architecture, combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) layers. The model has been trained on a labeled dataset to identify shoplifting behavior in video footage.

## Customization

You can customize the model further by training it on your own dataset or adjusting hyperparameters in the code. Refer to the model architecture in `shoplifting_model.py` for more details.

## Acknowledgments

- The model architecture is inspired by [Spatio-Temporal ConvLSTM for Crash Prediction](https://towardsdatascience.com/spatial-temporal-convlstm-for-crash-prediction-411909ed2cfa).

## Credits

This project was developed by:
- Jashwant Singh Yadav
- Durgesh Chaubey
- Adarsh Shukla

Guided by:
- Dr. Rohit Gupta

Special thanks to the contributors for their valuable contributions to the project.

Feel free to reach out to the developers for any questions or issues related to the project.

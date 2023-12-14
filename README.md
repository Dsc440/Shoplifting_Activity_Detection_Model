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
git clone https://github.com/Dsc440/Shoplifting_Activity_Detection_Model.git
```

2. Open the `Test_model.py` file and update the `input_video_path` variable with the path to the CCTV footage you want to test.

3. Save the changes.

### Running the Model

Execute the following command to run the model testing script:

```bash
python Test_model.py
```

The script will load the pre-trained CNLSTM2D model and apply it to the specified CCTV footage to detect shoplifting activity.

## Model Details

The shoplifting detection model is based on a CNLSTM2D architecture, combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) layers. The model has been trained on a labeled dataset to identify shoplifting behavior in video footage.

## Customization

You can customize the model further by training it on your own dataset. If you wish to use the dataset we used for training, you can find it [here](https://data.mendeley.com/datasets/r3yjf35hzr/1). The dataset consists of labeled examples of shoplifting activity in CCTV footage.

To train the model on your own dataset, follow these steps:

1. Download the dataset from [this link](https://data.mendeley.com/datasets/r3yjf35hzr/1).

2. Preprocess your dataset to ensure it has the required structure for training. You may need to adapt the data loading and preprocessing steps in the code accordingly.

3. Train the model using the updated dataset. You can adjust hyperparameters, such as the number of epochs, learning rate, and model architecture, to fine-tune the performance based on your specific data.

Refer to the `convlstmtry.ipynb` file for details on the model architecture and training process.

## Model Training and Evaluation

### Loss Plot

During the training of the shoplifting detection model, the loss function is monitored and recorded at each epoch. The loss plot provides insights into how well the model is learning from the training data and whether it is converging to a good solution.

![Loss Plot](https://github.com/Dsc440/Shoplifting_Activity_Detection_Model/blob/main/loss_plot.png?raw=true)

### Performance Metrics

The performance of the model is evaluated using various metrics, including but not limited to:

- **Accuracy:** Measures the overall correctness of the model's predictions.
- **Precision:** Indicates the ratio of true positive predictions to the total positive predictions, emphasizing the accuracy of positive predictions.
- **Recall:** Measures the ratio of true positive predictions to the total actual positives, highlighting the ability to capture positive instances.
- **F1 Score:** The harmonic mean of precision and recall, providing a balanced measure of a model's performance.

For a detailed understanding of model performance, consider reviewing the confusion matrix, which provides a breakdown of true positive, true negative, false positive, and false negative predictions.

### Confusion Matrix

The confusion matrix for the shoplifting detection model is presented below:

![Confusion Matrix](https://github.com/Dsc440/Shoplifting_Activity_Detection_Model/blob/main/conf_matrix%20(1).png?raw=true)

**Confusion Matrix:**

|                | Predicted Negative | Predicted Positive |
| -------------- | ------------------ | ------------------ |
| **Actual Negative** | True Negative | False Positive |
| **Actual Positive** | False Negative | True Positive |

The confusion matrix helps in understanding the distribution of model predictions across different classes. It is a valuable tool for assessing the model's performance and identifying areas of improvement.

**Model Metrics:**
- Accuracy: 0.70
- F1 Score: 0.705

These metrics are calculated during the evaluation phase.
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

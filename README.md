# Character Prediction using Multi-Layer Perceptron (MLP)

This project explores the application of Multi-Layer Perceptron (MLP) for character prediction tasks. The model is trained on the Paul Graham essays dataset, demonstrating the effectiveness of MLPs in capturing sequential patterns in text data.

## Dataset

The dataset consists of essays authored by Paul Graham, providing diverse linguistic content for training the MLP model.

## Model Architecture

The MLP model architecture is designed to process sequential data by transforming input sequences of characters into meaningful representations through multiple hidden layers. The model learns to predict the next character in a sequence based on the preceding characters.

## Implementation

The MLP model is implemented using Python and the TensorFlow framework. The training process involves feeding sequences of characters from the dataset to the MLP model and optimizing its parameters using backpropagation to minimize prediction errors.

## Next Character Prediction

Once trained, the MLP model can generate text by predicting the next character in a sequence given an input prompt. A Streamlit application is developed to provide a user-friendly interface, allowing users to input text and receive predictions for the next k characters.

## Visualization

Although MLPs typically do not produce embeddings like recurrent models, the model's performance can be evaluated through visualization of its training and validation metrics. Additionally, visualization techniques such as confusion matrices can be used to analyze model predictions.

## Future Work

- Experiment with different MLP architectures, including variations in the number of layers and units per layer, to optimize model performance.
- Explore techniques for incorporating attention mechanisms or recurrent connections into MLPs to improve their ability to capture long-range dependencies in text data.
- Extend the application to support additional datasets and languages, enabling broader applications of character prediction using MLPs.

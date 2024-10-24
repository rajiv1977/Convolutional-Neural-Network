## CNNModel 

The CNNModel class implements a Convolutional Neural Network (CNN) model. It inherits from the abstract Model class and provides the functionality to build, train, 
and make predictions with a CNN architecture. This model is designed to classify images using convolutional layers, activation layers, pooling layers, and fully connected layers.
The model follows a typical CNN architecture, combining convolutional layers to extract spatial hierarchies of features, activation layers to introduce non-linearities, pooling 
layers to reduce dimensions and improve computational efficiency, and fully connected layers to perform the final classification based on learned features. 
The output layer provides the final class probabilities, making it suitable for multi-class classification tasks.

# Features
* Layer Architecture: The model includes multiple convolutional layers, activation layers (ReLU), pooling layers, a flatten layer, and fully connected layers.
* Training: Implements the training process with forward and backward passes, using a specified loss function (Focal Loss).
* Prediction: Provides functionality to make predictions on input data after the model has been trained.

# Getting Started

## Prerequisite (Windows or Unix):
* Conan
* Matlab (2023 or above)
* Visual Studio 2022 (Win)

## Build (Window):
* Run **run_debug.bat** or **run_release.bat**

## Training the Model
Use the Train method to train the model with input data and corresponding labels. The data should be provided in a 4D vector format:
```Matlab
std::vector<std::vector<std::vector<std::vector<float>>>> trainingData; // Your input images
std::vector<std::vector<float>> trainingLabels; // Corresponding labels

cnnModel.Train(trainingData, trainingLabels); // Train the model
```

## Training Process
The model is trained using a specified number of epochs and a learning rate (default: 0.01).
It performs a forward pass to get predictions, computes the loss using Focal Loss, and then performs a backward pass to update weights through backpropagation.

## Sampel test:
```Matlab
#include <iostream>
#include <vector>
#include <memory>
#include "CNNModel.h"
#include "ConvolutionalLayer.h"
#include "ReLUActivationLayer.h"
#include "DropoutLayer.h"
#include "PoolingLayer.h"
#include "FlattenLayer.h"
#include "FullyConnectedLayer.h"
#include "CrossEntropyLoss.h"

/// @brief Example test function for CNNModel.
/// This function sets up a basic CNN architecture and performs a forward and backward pass.
void TestCNNModel()
{
    // Create mock data (for simplicity, use small random values)
    std::vector<std::vector<std::vector<std::vector<float>>>> data = {
        {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}}}; // 1 sample, 1 channel, 4x4 image
    std::vector<std::vector<float>> labels = {{1, 0, 0}};  // One-hot encoded label for 3 classes

    // Define CNN layers (using a simplified model for testing)
    std::vector<std::unique_ptr<Layer>> layers;

    // Convolutional layer: 1 channel input, 32 filters, kernel size 5x5
    layers.push_back(std::make_unique<ConvolutionalLayer>(1, 32, 5, 1));
    layers.push_back(std::make_unique<ReLUActivationLayer>());

    // Pooling layer: 2x2 pooling
    layers.push_back(std::make_unique<PoolingLayer>(2, 2));

    // Another convolutional layer: 32 input channels, 64 filters
    layers.push_back(std::make_unique<ConvolutionalLayer>(32, 64, 5, 1));
    layers.push_back(std::make_unique<ReLUActivationLayer>());

    // Pooling layer: 2x2 pooling
    layers.push_back(std::make_unique<PoolingLayer>(2, 2));

    // Flatten the output from the convolutional layers
    layers.push_back(std::make_unique<FlattenLayer>());

    // Fully connected layer: input size, output size (example: 3136 input to 10 outputs)
    layers.push_back(std::make_unique<FullyConnectedLayer>(3136, 10));
    layers.push_back(std::make_unique<ReLUActivationLayer>());

    // Dropout layer with 50% dropout
    layers.push_back(std::make_unique<DropoutLayer>(0.5));

    // Output layer: fully connected to the number of classes (e.g., 10)
    layers.push_back(std::make_unique<FullyConnectedLayer>(10, 3));  // Assume 3 classes for simplicity
    layers.push_back(std::make_unique<OutputLayer>());  // Softmax-like output

    // Initialize the CNNModel with the layers and loss type (CrossEntropyLoss used here)
    CNNModel model(std::move(layers), CNNModel::LossType::FOCAL);

    // Train the model on mock data
    std::cout << "Starting training..." << std::endl;
    model.Train(data, labels);

    // Validate the model on the same mock data (for simplicity)
    std::cout << "Validating model..." << std::endl;
    auto [valLoss, accuracy] = model.Validate(data, labels);

    std::cout << "Validation Loss: " << valLoss << ", Accuracy: " << accuracy << std::endl;
}

int main()
{
    // Run the test function for CNNModel
    TestCNNModel();

    return 0;
}

```

## Referece: 
OpenAI. "ChatGPT." *OpenAI*, 2024, https://openai.com/chatgpt.

Written with the help of AI. 


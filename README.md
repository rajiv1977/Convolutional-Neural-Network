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

## Building the Model
To create an instance of the CNNModel, simply instantiate it:
```Matlab
CNNModel cnnModel; // Create a CNNModel object
```

## Training the Model
Use the Train method to train the model with input data and corresponding labels. The data should be provided in a 4D vector format:
```Matlab
std::vector<std::vector<std::vector<std::vector<float>>>> trainingData; // Your input images
std::vector<std::vector<float>> trainingLabels; // Corresponding labels

cnnModel.Train(trainingData, trainingLabels); // Train the model
```

# Model Architecture for the example used
## Diagram
```Matlab
Input -> Conv1 -> ReLU -> Pool1 -> Conv2 -> ReLU -> Pool2 -> Flatten -> Fully Connected Layer -> ReLU -> Output Layer
```
![image](https://github.com/user-attachments/assets/3c6bba4c-4b82-4eb6-8d7a-1285e267709c)


# Description of the example used
1. Input Layer: Accepts input images with shape (28, 28, 1) for grayscale images. The input layer can be adjusted for different dimensions.

2. Convolutional Layer 1 (Conv1):
    - Input Channels: 1 (grayscale image)
    - Output Channels: 32
    - Kernel Size: 5x5
    - Stride: 1
    - Applies 32 filters to the input to extract features.

3. ReLU Activation Layer: Applies the Rectified Linear Unit activation function to introduce non-linearity.

4.  Pooling Layer 1 (Pool1):
      - Type: MaxPooling
      - Kernel Size: 2x2
      - Reduces the spatial dimensions (height and width) while retaining the most important features.

5. Convolutional Layer 2 (Conv2):
      - Input Channels: 32
      - Output Channels: 64
      - Kernel Size: 5x5
      - Stride: 1
      - Further extracts features from the output of the previous pooling layer.

6.  ReLU Activation Layer: Applies the activation function again.

7.  Pooling Layer 2 (Pool2): Another max pooling operation to reduce the dimensionality of the output.

8.  Flatten Layer: Converts the 3D feature maps into a 1D vector to prepare for the fully connected layers.

9.  Fully Connected Layer:
      - Input Size: 64 * 4 * 4 (based on the output size from pooling)
      - Output Size: 1024
      - Fully connects to the previous layer's output to integrate features for classification.

10.  ReLU Activation Layer: Applies the activation function to introduce non-linearity.

11.  Output Layer:
      - Output Size: 10
      - Computes the probabilities for each of the 10 classes in the classification task.

## Training Process
The model is trained using a specified number of epochs (default: 10) and a learning rate (default: 0.01).
It performs a forward pass to get predictions, computes the loss using Focal Loss, and then performs a backward pass to update weights through backpropagation.

## Conclusion
The CNNModel class provides a robust framework for building and training convolutional neural networks. It is suitable for image classification tasks and can be easily extended for more
complex architectures or different types of data.

## Sampel test:
```Matlab
#include "CNNModel.h"

int main()
{
    CNNModel cnn;

    // Example data: (num_samples, channels, height, width)
    std::vector<std::vector<std::vector<std::vector<float>>>> trainingData = {
        // Add your training data here
    };

    std::vector<std::vector<float>> trainingLabels = {
        // Add your corresponding labels here
    };

    // Train the model
    cnn.Train(trainingData, trainingLabels);

    // Example test data
    std::vector<std::vector<std::vector<float>>> testData = {
        // Add your test data here
    };

    // Predict
    for (const auto& sample : testData)
    {
        std::vector<float> prediction = cnn.Predict(sample);
        // Process the prediction
    }

    return 0;
}
```

## Referece: 
OpenAI. "ChatGPT." *OpenAI*, 2024, https://openai.com/chatgpt.

Written with the help of AI. 


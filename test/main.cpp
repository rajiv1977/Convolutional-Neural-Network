#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <filesystem>         // Include the filesystem library for handling directories and files
#include <iterator>
#include <iostream>
#include "ImageLoader.h"
#include <iomanip> // for std::setw
#include <opencv2/opencv.hpp> // Include OpenCV header for image loading
#include <vector>

#include "CNNModel.h"

void printDigitRepresentationsAndLabels(std::vector<std::vector<std::vector<std::vector<float>>>>& data,
                                        std::vector<std::vector<float>>&                           oneHotLabels)
{

    int index = 0;
    for (const auto& digitRepresentations : data)
    {
        std::cout << "Digit " << index << ":\n";
        for (const auto& digits : digitRepresentations)
        {
            for (const auto& row : digits)
            {
                std::copy(row.begin(), row.end(), std::ostream_iterator<int>(std::cout, " "));
                std::cout << "\n";
            }
        }
        std::cout << "\n\n";
        index++;
    }

    std::cout << "\n\n";

    for (auto& label : oneHotLabels)
    {
        std::cout << "One-hot encoded label: ";
        std::copy(label.begin(), label.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << "\n";
    }
}

int main()
{
    // Generate dummy training data
    size_t numImagesToLoad = 50; // Specify the number of images to load from each folder
    size_t numClasses      = 10; // Number of classes (digits 0-9)
    size_t numSamples      = numImagesToLoad * numClasses;
    size_t numChannels     = 1;     // For grayscale images
    size_t height          = 120;   // Image height
    size_t width           = 120;   // Image width
    float  epochs          = 5000;  // Number of epochs for training
    float  stopCriteria    = 0.001; // Stopping criteria for training
    float  learningRate    = 0.01;  // Learning rate for weight updates
    float  batchSize       = 1;     // Size of training batches
    float  dropoutRate     = 0.5f;  // drop out rate

    // Set the dataset path (make sure this path is valid on your system)
    std::string datasetPath = "E:\\AI_DATA\\TrainAndValidateData";

    // Create an instance of ImageLoader
    ImageLoader imageLoader(datasetPath);

    //  Selecting the inintial control factors
    imageLoader.UpdateSettings(true, true, {height, width}, false, numClasses);
    auto loadedImages = imageLoader.GetImagesFromFolder(numImagesToLoad);
    auto trainData    = loadedImages.images;
    auto trainLabels  = loadedImages.labels;
    height            = trainData[0][0].size(); 
    width             = trainData[0][0][0].size();

    // Define the CNN architecture
    std::vector<std::unique_ptr<Layer>> layers;

    std::cout << std::format("Building CNN model") << std::endl;

    //// Build model
 
    // Input layer: (1, 40,  40)
    layers.emplace_back(
        std::make_unique<ConvolutionalLayer>(1, 32, 5, 1));       // 1 input channel, 32 filters, 5x5 kernel, stride 1
    layers.emplace_back(std::make_unique<ReLUActivationLayer>()); // ReLU activation
    layers.emplace_back(std::make_unique<DropoutLayer>(dropoutRate));             // Dropout with 0.5 rate
    layers.emplace_back(std::make_unique<PoolingLayer>(2, PoolingType::AVERAGE)); // 2x2 Average Pooling

    layers.emplace_back(
        std::make_unique<ConvolutionalLayer>(32, 64, 5, 1));      // 32 input channels, 64 filters, 5x5 kernel, stride 1
    layers.emplace_back(std::make_unique<ReLUActivationLayer>()); // ReLU activation
    layers.emplace_back(std::make_unique<DropoutLayer>(dropoutRate));             // Dropout with 0.5 rate
    layers.emplace_back(std::make_unique<PoolingLayer>(2, PoolingType::AVERAGE)); // 2x2 Average Pooling

    // Flatten layer to connect the convolutional layers to the fully connected layer
    layers.emplace_back(std::make_unique<FlattenLayer>()); // Flatten the output from the pooling layer

    // Fully connected layer
    layers.emplace_back(std::make_unique<FullyConnectedLayer>(
        7 * 7 * 64, 1024)); // Input size after pooling is 7x7 for 64 channels, flatten to 3136
    layers.emplace_back(std::make_unique<ReLUActivationLayer>());     // ReLU activation for the fully connected layer
    layers.emplace_back(std::make_unique<DropoutLayer>(dropoutRate)); // Dropout with 0.5 rate

    // Output layer
    layers.emplace_back(
        std::make_unique<FullyConnectedLayer>(1024, numClasses)); // 10 output classes for classification
    layers.emplace_back(std::make_unique<OutputLayer>()); // Output layer to compute probabilities for each class

    // Create the CNN model
    CNNModel cnnModel(std::move(layers), CNNModel::LossType::MEAN_SQUARED_ERROR); // Move layers to avoid copying

    // Setting up the initial control parameters
    cnnModel.SetTrainingParameters(epochs, stopCriteria, learningRate); 

    // Train the model
    cnnModel.Train(trainData, trainLabels);

    // Validate the model
    auto valData                = loadedImages.images;
    auto valLabels              = loadedImages.labels;
    auto [valLoss, valAccuracy] = cnnModel.Validate(valData, valLabels);

    std::cout << "Validation Loss: " << valLoss << ", Validation Accuracy: " << valAccuracy << std::endl;

    return 0;
}

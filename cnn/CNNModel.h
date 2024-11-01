#pragma once

#include <algorithm>
#include <format>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "ConvolutionalLayer.h"
#include "CrossEntropyLoss.h" 
#include "DropoutLayer.h" 
#include "FlattenLayer.h" 
#include "FocalLoss.h" 
#include "FullyConnectedLayer.h" 
#include "L2Loss.h"  
#include "Layer.h" 
#include "Loss.h" 
#include "MeanSquaredError.h" 
#include "Model.h" 
#include "OutputLayer.h" 
#include "PoolingLayer.h"  
#include "ReLUActivationLayer.h" 

/// @class CNNModel
/// @brief A specific derived class representing a Convolutional Neural Network (CNN) model.
class CNNModel : public Model
{
  public:
    /// @brief Enum for specifying the type of loss function.
    enum class LossType
    {
        FOCAL,
        L2,
        MEAN_SQUARED_ERROR // Added MeanSquaredError type
    };

    /// @brief Constructor for the CNNModel class.
    /// Initializes the CNN architecture and loss calculator based on the specified loss type.
    /// @param lossType The type of loss function to use for training.
    CNNModel(std::vector<std::unique_ptr<Layer>> layers, LossType lossType = LossType::FOCAL)
        : layers_(std::move(layers))
    {
        // Initialize lossCalculator_ based on the chosen type
        switch (lossType)
        {
            case LossType::FOCAL:
                lossCalculator_ = std::make_unique<FocalLoss>();
                break;
            case LossType::L2:
                lossCalculator_ = std::make_unique<L2Loss>();
                break;
            case LossType::MEAN_SQUARED_ERROR: // Handle MeanSquaredError case
                lossCalculator_ = std::make_unique<MeanSquaredError>();
                break;
            default:
                throw std::invalid_argument("Unsupported loss type.");
        }

        // Initialize CNN architecture
        Build();
    }

    /// @brief Destructor for the CNNModel class.
    /// Cleans up resources to prevent memory leaks.
    ~CNNModel() override = default;

    /// @brief Builds the CNN model architecture.
    /// Constructs the layers of the CNN, specifying the parameters for each layer.
    void Build() override
    {
    }

    /// @brief Trains the CNN model with input data and corresponding labels.
    /// This function implements the training process for the CNN, including forward and backward passes.
    /// @param [in] data A 4D vector representing the input images of shape (num_samples, channels, height, width).
    /// @param [in] labels A 2D vector representing the corresponding labels for the input data.
    void Train(const std::vector<std::vector<std::vector<std::vector<float>>>>& data,
               const std::vector<std::vector<float>>&                           labels) override
    {
        const float  initialLearningRate      = learningRate_; // Initial learning rate for weight updates
        const int    numEpochs                = epochs_;       // Maximum number of epochs for training
        const size_t numSamples               = data.size();
        const int    patience                 = stopCriteria_; // Early stopping patience
        float        bestLoss                 = std::numeric_limits<float>::max();
        int          epochsWithoutImprovement = 0;

        for (int epoch = 0; epoch < numEpochs; ++epoch)
        {
            std::cout << "Epoch " << epoch + 1 << "/" << numEpochs << std::endl;
            std::cout << "\n" << std::endl;

            float epochLoss = 0.0f;

            for (size_t i = 0; i < numSamples; ++i)
            {
                std::cout << "Processing sample " << i + 1 << "/" << numSamples << std::endl;

                // Forward pass: obtain predictions
                auto predictions = Predict(data[i]);

                // Compute loss using the selected loss function
                float loss = lossCalculator_->ComputeLoss(labels[i], predictions); // Update based on chosen loss
                epochLoss += loss;

                std::cout << "Loss for sample " << i + 1 << ": " << loss << std::endl;

                // Backpropagation: update weights
                Backpropagation(predictions, labels[i], initialLearningRate);
            }

            // Average loss for the epoch
            epochLoss /= numSamples;
            std::cout << "Average Loss for Epoch " << epoch + 1 << ": " << epochLoss << std::endl;

            // Check for early stopping
            if (epochLoss < bestLoss)
            {
                bestLoss                 = epochLoss;
                epochsWithoutImprovement = 0;
                // Optionally reduce learning rate here if necessary
            }
            else
            {
                epochsWithoutImprovement++;
                if (epochsWithoutImprovement >= patience)
                {
                    std::cout << "Early stopping triggered." << std::endl;
                    break; // Stop training if no improvement
                }
            }
            std::cout << "\n" << std::endl;
        }
        std::cout << "Training completed." << std::endl;
    }

    /// @brief Validates the CNN model using the provided validation data and labels.
    /// This function calculates the average validation loss and accuracy over the validation dataset.
    /// @param [in] valData A 4D vector representing the input images of shape (num_samples, channels, height, width)
    /// for validation.
    /// @param [in] valLabels A 2D vector representing the corresponding labels for the validation data.
    /// @return A pair containing the average validation loss and accuracy.
    std::pair<float, float> Validate(const std::vector<std::vector<std::vector<std::vector<float>>>>& valData,
                                     const std::vector<std::vector<float>>&                           valLabels)
    {
        float epochValLoss       = 0.0f; // To accumulate total validation loss
        int   correctPredictions = 0;    // To count correct predictions

        // Iterate over all validation samples
        for (size_t i = 0; i < valData.size(); ++i)
        {
            // Get predictions for the current validation sample
            auto valPredictions = Predict(valData[i]);

            // Get the true classifications
            auto valTruth = valLabels[i];

            // Print the true classifications
            std::cout << "truth" << std::endl;
            std::copy(valTruth.begin(), valTruth.end(), std::ostream_iterator<float>(std::cout, " \t"));
            std::cout << "\n" << std::endl; 

            std::cout << "predicted" << std::endl;
            std::copy(valPredictions.begin(), valPredictions.end(), std::ostream_iterator<float>(std::cout, " \t"));
            std::cout << "\n" << std::endl; 

            // Calculate loss for the current sample
            float valLoss = lossCalculator_->ComputeLoss(valLabels[i], valPredictions);

            epochValLoss += valLoss; // Accumulate loss for averaging

            // Determine the predicted label (index of the max prediction)
            int predictedLabel =
                std::distance(valPredictions.begin(), std::max_element(valPredictions.begin(), valPredictions.end()));
            // Determine the true label (index of the max in one-hot encoded true labels)
            int trueLabel =
                std::distance(valLabels[i].begin(), std::max_element(valLabels[i].begin(), valLabels[i].end()));

            // Increment correct predictions if the predicted label matches the true label
            if (predictedLabel == trueLabel)
            {
                correctPredictions++;
            }

            // Output the loss for the current validation sample
            std::cout << "Loss for validation sample " << i + 1 << ": " << valLoss << std::endl;
        }

        // Average validation loss
        epochValLoss /= valData.size();
        // Calculate accuracy as the ratio of correct predictions to total samples
        float accuracy = static_cast<float>(correctPredictions) / valData.size();

        // Output average validation loss and accuracy
        std::cout << "Average Validation Loss: " << epochValLoss << ", Accuracy: " << accuracy << std::endl;

        // Return both average validation loss and accuracy as a pair
        return {epochValLoss, accuracy};
    }

    /// @brief Makes predictions using the trained CNN model.
    /// This function performs a forward pass through all layers of the model to generate predictions.
    /// @param [in] input A 4D tensor representing the input data of shape (channels, height, width).
    /// @return A vector of floats representing the predicted output for each class.
    std::vector<float> Predict(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        std::cout << "Making predictions with CNN model." << std::endl;

        // Forward pass through all layers
        std::vector<std::vector<std::vector<float>>> output = input; // Start with the input

        for (const auto& layer : layers_)
        {
            output = layer->Forward(output); // Call the forward method of each layer
        }

        // Assuming the output tensor has the structure:
        if (!output.empty() && !output[0].empty())
        {
            return output[0][0]; // Return the output of the first sample
        }
        else
        {
            std::cerr << "Error: Output does not contain valid logits." << std::endl;
            return {}; // Return an empty vector on error
        }
    }

    std::vector<float> GetAllWeights() const
    {
        std::vector<float> weights;
        for (const auto& layer : layers_)
        {
            auto layerWeights = layer->GetAllWeights(); // Assuming each layer has a method to return its weights
            weights.insert(weights.end(), layerWeights.begin(), layerWeights.end());
        }
        return weights;
    }

  private:
    std::vector<std::unique_ptr<Layer>> layers_;         // Vector to hold the layers of the CNN
    std::unique_ptr<Loss>               lossCalculator_; // Pointer to Loss calculator

    /// @brief Performs backpropagation to update weights based on the loss gradient.
    /// This function computes the loss gradient and propagates it backward through the network.
    /// @param [in] predictions A vector of floats representing the predicted outputs from the model.
    /// @param [in] trueLabels A vector of floats representing the ground truth labels corresponding to the predictions.
    /// @param [in] learningRate A float value representing the learning rate for weight updates.
    void
        Backpropagation(const std::vector<float>& predictions, const std::vector<float>& trueLabels, float learningRate)
    {
        // Compute the loss gradient based on the predictions and true labels
        std::vector<float> lossGradient = ComputeLossGradient(predictions, trueLabels);

        // Convert the 1D lossGradient into a 3D format to match currentGradient
        std::vector<std::vector<std::vector<float>>> currentGradient(1,
                                                                     std::vector<std::vector<float>>(1, lossGradient));

        // Iterate through layers in reverse order to calculate gradients
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        {
            currentGradient = (*it)->Backward(currentGradient, learningRate); // Call backward method of each layer
        }
    }

    /// @brief Computes the loss gradient based on predictions and true labels.
    /// @param [in] predictions A vector of floats representing the predicted outputs from the model.
    /// @param [in] trueLabels A vector of floats representing the ground truth labels corresponding to the predictions.
    /// @return A vector of floats representing the computed loss gradient.
    std::vector<float> ComputeLossGradient(const std::vector<float>& predictions, const std::vector<float>& trueLabels)
    {
        // Use the loss calculator to compute gradients
        return lossCalculator_->ComputeGradient(predictions, trueLabels); // Use the specific loss to compute gradients
    }
};

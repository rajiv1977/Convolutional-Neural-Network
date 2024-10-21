#pragma once

#include <iostream>
#include <vector>

#include "Loss.h"

/// @class Model
/// @brief An abstract base class for machine learning models.
/// This class defines the essential methods that any derived machine learning model
/// should implement. It provides a blueprint for building models, training them with
/// data, and making predictions based on input data.
class Model
{
  public:
    /// @brief Sets the training parameters for the model.
    /// This method allows users to configure the training process by specifying the
    /// number of epochs, the stopping criteria, and other parameters as needed.
    ///
    /// @param epochs The number of training epochs.
    /// @param stopCriteria The criteria for stopping the training (e.g., loss threshold, validation accuracy).
    /// @param learningRate The learning rate used in the training process.
    /// @param batchSize The size of the training batches (added for CNN).
    virtual void SetTrainingParameters(int epochs, float stopCriteria, float learningRate, int batchSize = 1)
    {
        epochs_       = epochs;       // Number of epochs for training
        stopCriteria_ = stopCriteria; // Stopping criteria for training
        learningRate_ = learningRate; // Learning rate for weight updates
        batchSize_    = batchSize;    // Size of training batches
    }

    /// @brief Builds the machine learning model.
    /// This method should be overridden by derived classes to implement the specific
    /// logic required for building the model architecture.
    virtual void Build() = 0;

    /// @brief Trains the model using the provided RGB data and corresponding labels.
    /// This method takes in a 4D tensor representing RGB images and a 2D tensor for
    /// one-hot encoded labels, and trains the model accordingly.
    ///
    /// @param [in] data A 4D tensor of shape (num_samples, channels, height, width) representing the RGB images.
    /// @param [in] labels A 2D tensor of shape (num_samples, num_classes) for one-hot encoded labels.
    virtual void Train(
        const std::vector<std::vector<std::vector<std::vector<float>>>>& data, // (num_samples, channels, height, width)
        const std::vector<std::vector<float>>&                           labels) = 0;

    /// @brief Validates the model using the provided validation data and labels.
    /// This method should be overridden by derived classes to implement the specific
    /// logic required for validation.
    ///
    /// @param [in] data A 4D tensor of shape (num_samples, channels, height, width) representing the validation RGB
    /// images.
    /// @param [in] labels A 2D tensor of shape (num_samples, num_classes) for one-hot encoded labels.
    /// @return The average validation loss and accuracy as a pair.
    virtual std::pair<float, float> Validate(
        const std::vector<std::vector<std::vector<std::vector<float>>>>& data, // (num_samples, channels, height, width)
        const std::vector<std::vector<float>>&                           labels) = 0;

    /// @brief Makes predictions based on the input data.
    /// This method takes a 3D tensor representing an RGB image and outputs a vector
    /// containing the predicted results.
    ///
    /// @param [in] input A 3D tensor of shape (channels, height, width) representing an RGB image.
    /// @return A vector containing the predicted output.
    virtual std::vector<float> Predict(const std::vector<std::vector<std::vector<float>>>& input) = 0;

    /// @brief Sets the loss calculator for the model.
    /// This method allows the model to use a specific loss function for training and validation.
    ///
    /// @param [in] lossCalculator Pointer to a loss calculator object.
    virtual void SetLossCalculator(std::shared_ptr<Loss>& lossCalculator)
    {
        lossCalculator_ = lossCalculator; // Set the loss calculator
    }

    /// Virtual destructor for proper cleanup
    virtual ~Model() = default; // Ensures proper cleanup of derived classes

  protected:
    int             epochs_;         // Number of epochs for training
    float           stopCriteria_;   // Stopping criteria for training
    float           learningRate_;   // Learning rate for training
    int             batchSize_;      // Batch size for training (added for CNN)
    std::shared_ptr<Loss> lossCalculator_; // Use shared pointer for loss function
};

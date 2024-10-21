#pragma once

#include <random>
#include <vector>
#include "Layer.h"

/// @class DropoutLayer
/// @brief A class representing a dropout layer in a neural network.
///
class DropoutLayer : public Layer
{
  public:
    /// @brief Constructor for the DropoutLayer.
    /// Initializes the dropout layer with a specified dropout rate.
    ///
    /// @param rate A float representing the probability of setting a unit to zero (dropout rate).
    DropoutLayer(float rate)
        : dropoutRate_(rate)
        , isTraining_(true)
    {
    }

    /// @brief Forward pass through the Dropout layer.
    /// During training, sets a fraction of the input units to zero.
    /// During inference, passes the input through unchanged.
    ///
    /// @param input A 3D vector representing the input data (batch size, channels, height).
    /// @return A 3D vector representing the output after applying dropout.
    std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        output_ = input;            // Copy input to output
        mask_.resize(input.size()); // Resize mask to match the input size

        if (isTraining_)
        {
            std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
            std::mt19937                          generator; // Mersenne Twister RNG

            // Apply dropout and create mask
            for (size_t i = 0; i < output_.size(); ++i) // Iterate over batches
            {
                mask_[i].resize(output_[i].size());
                for (size_t j = 0; j < output_[i].size(); ++j) // Iterate over channels
                {
                    mask_[i][j].resize(output_[i][j].size());
                    for (size_t k = 0; k < output_[i][j].size(); ++k) // Iterate over height
                    {
                        if (distribution(generator) < dropoutRate_)
                        {
                            output_[i][j][k] = 0.0f; // Set unit to zero
                            mask_[i][j][k]   = 0.0f; // Mark unit as dropped in mask
                        }
                        else
                        {
                            mask_[i][j][k] = 1.0f; // Mark unit as kept in mask
                        }
                    }
                }
            }
        }

        return output_; // Return the output after applying dropout
    }

    /// @brief Backward pass through the Dropout layer.
    /// During the backward pass, the gradients are passed only for the units that were not dropped.
    ///
    /// @param outputGradient A 3D vector representing the gradients from the subsequent layer.
    /// @param learningRate A float representing the learning rate (unused for dropout).
    /// @return A 3D vector representing the gradients to be propagated back.
    std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& outputGradient, float learningRate) override
    {
        std::vector<std::vector<std::vector<float>>> gradInput = outputGradient;

        if (isTraining_) // During training, apply the mask to propagate gradients
        {
            for (size_t i = 0; i < gradInput.size(); ++i) // Iterate over batches
            {
                for (size_t j = 0; j < gradInput[i].size(); ++j) // Iterate over channels
                {
                    for (size_t k = 0; k < gradInput[i][j].size(); ++k) // Iterate over height
                    {
                        gradInput[i][j][k] *= mask_[i][j][k]; // Multiply gradient by mask
                    }
                }
            }
        }

        return gradInput; // Return gradients after applying mask
    }

    /// @brief Enables or disables training mode for the Dropout layer.
    ///
    /// @param isTraining A boolean indicating whether the layer should behave as training or inference.
    void SetTrainingMode(bool isTraining)
    {
        isTraining_ = isTraining; // Set training mode
    }

    /// @brief Update weights (noop for DropoutLayer as it has no weights).
    void UpdateWeights(float learningRate) override
    {
        // Dropout layer has no weights to update
    }

    /// @brief Get all weights (noop for DropoutLayer as it has no weights).
    /// @return An empty vector.
    std::vector<float> GetAllWeights() const override
    {
        return {}; // Dropout has no weights
    }

    /// @brief Compute L2 regularization loss (noop for DropoutLayer).
    /// @param l2RegularizationFactor The factor for scaling the regularization loss.
    /// @return 0.0f, as Dropout has no weights.
    float ComputeL2Regularization(float l2RegularizationFactor) const override
    {
        return 0.0f; // No weights, so no L2 regularization loss
    }

  private:
    float                                        dropoutRate_; // Probability of dropping out a unit
    bool                                         isTraining_;  // Flag to indicate if the layer is in training mode
    std::vector<std::vector<std::vector<float>>> output_;      // Output from the forward pass
    std::vector<std::vector<std::vector<float>>> mask_;        // Mask to store which units were dropped
};

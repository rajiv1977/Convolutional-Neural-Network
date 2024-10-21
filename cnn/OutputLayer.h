#pragma once

#include <iostream>
#include <cmath> // Include for exp function
#include <vector>

#include "Layer.h"

/// @class OutputLayer
/// @brief A specific layer representing the output layer of the CNN.
/// The OutputLayer takes logits from the previous layer and computes probabilities for each class.
class OutputLayer : public Layer
{
  public:
    /// @brief Default constructor for the OutputLayer class.
    OutputLayer() = default;

    /// @brief Forward pass for the OutputLayer.
    /// This function takes the logits as input and applies the softmax function to compute probabilities.
    ///
    /// @param [in] input A 3D tensor representing the input logits (batch_size, channels, logits).
    /// @return A 3D tensor representing the output probabilities (batch_size, channels, probabilities).
    std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        std::vector<std::vector<std::vector<float>>> output; // Initialize output tensor

        // Iterate over each sample in the batch
        for (const auto& sample : input)
        {
            std::vector<float> logits        = sample[0];       // Assuming the logits are passed as the first element
            std::vector<float> probabilities = Softmax(logits); // Apply softmax to get probabilities

            // Wrap probabilities in a 3D structure
            output.push_back({probabilities}); // Store the output in a 3D tensor format
        }

        return output; // Return the output tensor
    }

    /// @brief Backward pass for the OutputLayer.
    /// This function scales the upstream gradient by the learning rate and returns it.
    ///
    /// @param [in] upstreamGradient The gradient coming from the previous layer.
    /// @param learningRate The learning rate used for scaling the gradient.
    /// @return A 3D tensor representing the gradient for the upstream layer.
    std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& upstreamGradient, float learningRate) override
    {
        std::vector<std::vector<std::vector<float>>> scaledGradient = upstreamGradient; // Initialize scaled gradient

        // Scale the upstream gradient by the learning rate
        for (auto& sample : scaledGradient)
        {
            for (auto& channel : sample)
            {
                for (auto& value : channel)
                {
                    value *= learningRate; // Scale each gradient by the learning rate
                }
            }
        }

        return scaledGradient; // Return the scaled gradient
    }


    /// @brief Updates the weights of the OutputLayer (no-op).
    /// This function does nothing since the output layer does not have weights.
    ///
    /// @param learningRate The learning rate for weight updates.
    void UpdateWeights(float learningRate) override
    {
        // No weights to update
    }

    std::vector<float> GetAllWeights() const override
    {
        return {};
    }

    float ComputeL2Regularization(float l2RegularizationFactor) const
    {
        return 0;
    }

  private:
    /// @brief Applies the softmax function to the input logits.
    ///
    /// @param [in] logits A vector of raw scores (logits).
    /// @return A vector of probabilities corresponding to the input scores.
    std::vector<float> Softmax(const std::vector<float>& logits)
    {
        std::vector<float> expScores(logits.size());
        float              sumExpScores = 0.0f;

        // Compute the exponential of each score and the sum of exponentials
        for (size_t i = 0; i < logits.size(); ++i)
        {
            expScores[i] = std::exp(logits[i]); // Use std::exp for better readability
            sumExpScores += expScores[i];       // Accumulate sum of exponentials
        }

        // Normalize the scores to get probabilities
        std::vector<float> probabilities(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
        {
            probabilities[i] = expScores[i] / sumExpScores; // Normalize to get probabilities
        }

        return probabilities; // Return the computed probabilities
    }
};

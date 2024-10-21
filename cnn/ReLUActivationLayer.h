#pragma once

#include <algorithm> // For std::max
#include <stdexcept> // For std::invalid_argument
#include <vector>

#include "Layer.h" // Include the base Layer class

/// @class ReLUActivationLayer
/// @brief A class representing the ReLU activation function.
/// This class implements the forward pass for the ReLU activation
/// function, allowing for element-wise activation of the input tensor.
class ReLUActivationLayer : public Layer
{
  public:
    /// @brief Performs the forward pass of the ReLU activation function.
    /// This function takes the input tensor and applies the ReLU activation
    /// function, setting all negative values to zero.
    ///
    /// @param input A 3D vector representing the input tensor (channels, height, width).
    /// @return A 3D vector representing the output tensor after applying ReLU activation.
    std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        // Validate input dimensions
        if (input.empty() || input[0].empty() || input[0][0].empty())
        {
            throw std::invalid_argument("Input tensor dimensions must be non-zero.");
        }

        int numChannels = input.size();       // Get the number of channels
        int height      = input[0].size();    // Get input height
        int width       = input[0][0].size(); // Get input width

        // Initialize output tensor
        std::vector<std::vector<std::vector<float>>> output(
            numChannels, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));

        // Apply ReLU activation
        for (int c = 0; c < numChannels; ++c) // Iterate over channels
        {
            for (int h = 0; h < height; ++h) // Iterate over height
            {
                for (int w = 0; w < width; ++w) // Iterate over width
                {
                    output[c][h][w] = std::max(0.0f, input[c][h][w]); // Apply ReLU
                }
            }
        }
        input_ = input; // Store input for backpropagation
        return output;  // Return the activated output tensor
    }

    /// @brief Performs the backward pass of the ReLU activation function.
    /// This function computes the gradient of the loss with respect to the input
    /// tensor by checking where the input was positive during the forward pass.
    /// The upstream gradients are scaled by the learning rate.
    ///
    /// @param upstreamGradient A 3D vector representing the upstream gradient (channels, height, width).
    /// @param learningRate The learning rate used to scale the upstream gradients.
    /// @return A 3D vector representing the gradient with respect to the input tensor.
    std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& upstreamGradient, float learningRate) override
    {
        // Validate upstream gradient dimensions
        if (upstreamGradient.size() != input_.size() || upstreamGradient[0].size() != input_[0].size() ||
            upstreamGradient[0][0].size() != input_[0][0].size())
        {
            throw std::invalid_argument("Upstream gradient dimensions do not match the input dimensions.");
        }

        int numChannels = upstreamGradient.size();       // Get number of channels
        int height      = upstreamGradient[0].size();    // Get height
        int width       = upstreamGradient[0][0].size(); // Get width

        // Initialize input gradient tensor
        std::vector<std::vector<std::vector<float>>> inputGradient(
            numChannels, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));

        for (int c = 0; c < numChannels; ++c) // Iterate over channels
        {
            for (int h = 0; h < height; ++h) // Iterate over height
            {
                for (int w = 0; w < width; ++w) // Iterate over width
                {
                    // Check if the input value was greater than zero during the forward pass
                    if (input_[c][h][w] > 0)
                    {
                        inputGradient[c][h][w] = upstreamGradient[c][h][w] * learningRate; // Scale by learning rate
                    }
                    // No need to set inputGradient[c][h][w] to zero, as it's already initialized to zero
                }
            }
        }

        return inputGradient; // Return the input gradient
    }

    /// @brief Updates the weights of the ReLU activation layer.
    /// Since ReLU does not have weights, this function is a no-op.
    ///
    /// @param learningRate The learning rate (not used).
    void UpdateWeights(float learningRate) override
    {
        // No weights to update for ReLU
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
    std::vector<std::vector<std::vector<float>>> input_; // Store input for backpropagation
};

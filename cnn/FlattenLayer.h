#pragma once

#include <stdexcept>
#include <vector>

#include "Layer.h"

/// @class FlattenLayer
/// @brief A class representing a flattening layer in a neural network.
///
class FlattenLayer : public Layer
{
  public:
    /// @brief Default constructor for the FlattenLayer class.
    FlattenLayer() = default;

    /// @brief Performs the forward pass of the flatten layer.
    /// This method takes a 3D input tensor and flattens it into a 3D output tensor
    /// with a single batch dimension.
    ///
    /// @param [in] input A 3D tensor representing the input data to the layer,
    ///                  with dimensions (channels, height, width).
    /// @return A 3D tensor representing the output data after flattening,
    ///         with dimensions (1, 1, channels * height * width).
    /// @throws std::invalid_argument If the input tensor is empty or has an incorrect shape.
    std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        // Check if input is empty
        if (input.empty() || input[0].empty() || input[0][0].empty())
        {
            throw std::invalid_argument("Input cannot be empty or incorrectly shaped.");
        }

        // Get dimensions of the input
        size_t channels = input.size();       // Number of channels (e.g., RGB)
        size_t height   = input[0].size();    // Height of the input
        size_t width    = input[0][0].size(); // Width of the input

        // Compute the size of the flattened output
        size_t flattenedSize = channels * height * width;

        // Create the output tensor with one batch dimension
        std::vector<std::vector<std::vector<float>>> output(
            1,
            std::vector<std::vector<float>>(1, std::vector<float>(flattenedSize))); // Output shape: 1 x 1 x (C * H * W)

        // Flatten the input tensor
        size_t index = 0;                     // Index for the output tensor
        for (size_t c = 0; c < channels; ++c) // Loop over channels
        {
            for (size_t h = 0; h < height; ++h) // Loop over height
            {
                for (size_t w = 0; w < width; ++w) // Loop over width
                {
                    output[0][0][index++] = input[c][h][w]; // Assign flattened values to output tensor
                }
            }
        }

        return output; // Return the flattened output
    }

    /// @brief Performs the backward pass of the flatten layer.
    /// This method takes a 3D upstream gradient tensor and reshapes it back to
    /// the original input tensor dimensions for the previous layer in the network.
    ///
    /// @param [in] upstreamGradient A 3D tensor representing the gradient of the loss
    ///                              with respect to the output of this layer,
    ///                              with dimensions (number of samples, 1, channels * height * width).
    /// @param [in] learningRate The learning rate for any potential updates during backpropagation.
    /// @return A 3D tensor representing the gradient of the loss with respect to
    ///         the input of this layer, with dimensions (number of samples, channels, height, width).
    std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& upstreamGradient, float learningRate) override
    {
        // Get dimensions from the upstream gradient
        size_t numSamples    = upstreamGradient.size();       // Number of samples
        size_t flattenedSize = upstreamGradient[0][0].size(); // Size of the flattened gradient

        // Assuming original input dimensions are known
        size_t channels = 1;             // Number of channels (e.g., RGB = 3)
        size_t height   = 1;             // Set this based on the original input height
        size_t width    = flattenedSize; // Set this based on the original input width

        // Ensure that the flattened size matches the expected size
        if (flattenedSize != channels * height * width)
        {
            throw std::invalid_argument("Flattened size does not match original dimensions.");
        }

        // Create the output gradient tensor
        std::vector<std::vector<std::vector<float>>> output(
            numSamples,
            std::vector<std::vector<float>>(channels,
                                            std::vector<float>(height * width))); // Output shape: N x C x (H * W)

        // Reshape the upstream gradient back to the original dimensions
        for (size_t sample = 0; sample < numSamples; ++sample) // Loop over each sample
        {
            for (size_t index = 0; index < flattenedSize; ++index) // Loop over flattened gradient
            {
                // Determine the original channel, height, and width from the flattened index
                size_t c = index / (height * width);           // Determine the channel
                size_t h = (index % (height * width)) / width; // Determine the height
                size_t w = (index % (height * width)) % width; // Determine the width

                // Assign values to output gradient
                output[sample][c][h * width + w] = upstreamGradient[sample][0][index];
            }
        }

        // Learning rate usage (if applicable)
        // Note: Since FlattenLayer does not have weights, this is just a placeholder for potential future use.

        return output; // Return the reshaped gradients
    }

    /// @brief Updates the weights of the flatten layer.
    /// This method is not applicable for the FlattenLayer as it does not have weights to update.
    ///
    /// @param [in] learningRate The learning rate for adjusting the weights during the update process.
    void UpdateWeights(float learningRate) override
    {
        // FlattenLayer does not have weights, so this function does nothing
    }

    std::vector<float> GetAllWeights() const override
    {
        return {};
    }

    float ComputeL2Regularization(float l2RegularizationFactor) const
    {
        return 0;
    }
};

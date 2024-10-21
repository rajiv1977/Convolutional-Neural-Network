#pragma once

#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Layer.h" // Include the base Layer class

/// @enum PoolingType
/// @brief Enum for selecting pooling types.
enum class PoolingType
{
    MAX,    // Max pooling
    AVERAGE // Average pooling
};

/// @class PoolingLayer
/// @brief A class representing a pooling layer in a CNN.
class PoolingLayer : public Layer // Inherit from Layer
{
  public:
    /// @brief Constructor for the PoolingLayer class.
    /// Initializes the pooling layer with the specified pooling size and type.
    ///
    /// @param poolSize The size of the pooling window.
    /// @param poolingType The type of pooling operation (max or average).
    PoolingLayer(int poolSize, PoolingType poolingType)
        : poolSize_(poolSize)
        , poolingType_(poolingType)
    {
    }

    /// @brief Performs the forward pass of the pooling layer.
    /// This function takes the input tensor and computes the output tensor using
    /// the specified pooling operation, reducing its spatial dimensions.
    ///
    /// @param input A 3D vector representing the input tensor (channels, height, width).
    /// @return A 3D vector representing the output tensor after the pooling operation.
    std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        // Validate input dimensions
        if (input.empty() || input[0].empty() || input[0][0].empty())
        {
            throw std::invalid_argument("Input tensor must not be empty.");
        }

        int numChannels = input.size();       // Get the number of channels
        int inputHeight = input[0].size();    // Get input height
        int inputWidth  = input[0][0].size(); // Get input width

        // Calculate output dimensions
        int outputHeight = (inputHeight + poolSize_ - 1) / poolSize_; // Adaptive height calculation
        int outputWidth  = (inputWidth + poolSize_ - 1) / poolSize_;  // Adaptive width calculation

        // Initialize output tensor and max indices for backpropagation
        std::vector<std::vector<std::vector<float>>> output(
            numChannels, std::vector<std::vector<float>>(outputHeight, std::vector<float>(outputWidth, 0.0f)));

        maxIndices_.resize(
            numChannels,
            std::vector<std::vector<std::pair<int, int>>>(outputHeight, std::vector<std::pair<int, int>>(outputWidth)));

        // Perform pooling based on the selected pooling type
        for (int c = 0; c < numChannels; ++c) // Iterate over channels
        {
            for (int h = 0; h < outputHeight; ++h) // Iterate over output height
            {
                for (int w = 0; w < outputWidth; ++w) // Iterate over output width
                {
                    // Initialize max value and sum for average pooling
                    float poolValue =
                        (poolingType_ == PoolingType::MAX) ? -std::numeric_limits<float>::infinity() : 0.0f;
                    int count = 0; // Counter for average pooling

                    // Perform pooling
                    for (int kh = 0; kh < poolSize_; ++kh) // Iterate over pooling height
                    {
                        for (int kw = 0; kw < poolSize_; ++kw) // Iterate over pooling width
                        {
                            // Calculate the input indices
                            int inputH = h * poolSize_ + kh;
                            int inputW = w * poolSize_ + kw;

                            // Ensure indices are within bounds
                            if (inputH < inputHeight && inputW < inputWidth)
                            {
                                float value = input[c][inputH][inputW];
                                if (poolingType_ == PoolingType::MAX)
                                {
                                    // Max pooling
                                    if (value > poolValue)
                                    {
                                        poolValue            = value;
                                        maxIndices_[c][h][w] = std::make_pair(kh, kw); // Store indices of max value
                                    }
                                }
                                else if (poolingType_ == PoolingType::AVERAGE)
                                {
                                    // Average pooling
                                    poolValue += value;
                                    count++;
                                }
                            }
                        }
                    }

                    // Store pooled value in output tensor
                    if (poolingType_ == PoolingType::MAX)
                    {
                        output[c][h][w] = poolValue; // Store max value
                    }
                    else if (poolingType_ == PoolingType::AVERAGE && count > 0)
                    {
                        output[c][h][w] = poolValue / count; // Store average value
                    }
                }
            }
        }

        return output; // Return the pooled output tensor
    }

    /// @brief Performs the backward pass of the pooling layer.
    /// This function computes the gradient of the loss with respect to the input
    /// tensor by using the stored indices from the forward pass and scales the gradient
    /// by the learning rate.
    ///
    /// @param upstreamGradient A 3D vector representing the upstream gradient (channels, height, width).
    /// @param learningRate The learning rate used for scaling the gradient.
    /// @return A 3D vector representing the gradient with respect to the input tensor.
    std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& upstreamGradient, float learningRate) override
    {
        // Check if upstreamGradient needs de-flattening
        const auto& internalGradient =
            (upstreamGradient.size() == 1 && upstreamGradient[0].size() == 1 && upstreamGradient[0][0].size() > 1)
                ? DeFlatten(upstreamGradient)
                : upstreamGradient;

        int numChannels  = internalGradient.size();
        int outputHeight = internalGradient[0].size();
        int outputWidth  = internalGradient[0][0].size();

        // Initialize input gradient tensor with zeros
        std::vector<std::vector<std::vector<float>>> inputGradient(
            numChannels,
            std::vector<std::vector<float>>(outputHeight * poolSize_,
                                            std::vector<float>(outputWidth * poolSize_, 0.0f)));

        // Backpropagate gradients
        for (int c = 0; c < numChannels; ++c) // Iterate over channels
        {
            for (int h = 0; h < outputHeight; ++h) // Iterate over output height
            {
                for (int w = 0; w < outputWidth; ++w) // Iterate over output width
                {
                    // Get the upstream gradient for this output position and scale it by learning rate
                    float grad = internalGradient[c][h][w] * learningRate;

                    // Get the indices of the max value from the forward pass
                    auto maxIndex = maxIndices_[c][h][w];
                    inputGradient[c][h * poolSize_ + maxIndex.first][w * poolSize_ + maxIndex.second] +=
                        grad; // Propagate the scaled gradient
                }
            }
        }

        return inputGradient; // Return the input gradient
    }

    /// @brief Updates the weights of the pooling layer.
    /// Since pooling layers do not have weights, this function is a no-op.
    ///
    /// @param learningRate The learning rate (not used).
    void UpdateWeights(float learningRate) override
    {
        // No weights to update for pooling layers
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
    int         poolSize_;    // Size of the pooling window
    PoolingType poolingType_; // Type of pooling operation (max or average)
    std::vector<std::vector<std::vector<std::pair<int, int>>>>
        maxIndices_; // Stores indices of max values for backpropagation

    /// @brief De-flattens the upstream gradient if required.
    /// This function reshapes the gradient from a flat structure back to the original pooling layer input dimensions.
    ///
    /// @param upstreamGradient A 3D vector representing the upstream gradient (channels, height, width).
    /// @return A 3D vector representing the de-flattened gradient with respect to the input tensor.
    std::vector<std::vector<std::vector<float>>>
        DeFlatten(const std::vector<std::vector<std::vector<float>>>& upstreamGradient)
    {
        int numChannels  = maxIndices_.size();       // Get the number of channels
        int outputHeight = maxIndices_[0].size();    // Get output height
        int outputWidth  = maxIndices_[0][0].size(); // Get output width

        std::vector<std::vector<std::vector<float>>> deflattenUpstreamGradient(
            numChannels, std::vector<std::vector<float>>(outputHeight, std::vector<float>(outputWidth, 0.0f)));

        size_t index = 0; // Index for the output tensor
        for (size_t c = 0; c < numChannels; ++c)
        {
            for (size_t h = 0; h < outputHeight; ++h)
            {
                for (size_t w = 0; w < outputWidth; ++w)
                {
                    deflattenUpstreamGradient[c][h][w] = upstreamGradient[0][0][index++];
                }
            }
        }

        return deflattenUpstreamGradient; // Return the de-flattened gradient
    }
};

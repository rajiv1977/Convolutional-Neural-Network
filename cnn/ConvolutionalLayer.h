#pragma once

#include <ctime>   // Include for time
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "Layer.h" // Include the base Layer class


 /// @class ConvolutionalLayer
 /// @brief Represents a convolutional layer in a neural network.
 /// 
 /// This class performs the forward and backward passes for a convolution operation,
 /// initializes weights and biases, and updates them during training.
class ConvolutionalLayer : public Layer
{
  public:
    /// @brief Constructs a ConvolutionalLayer with the specified parameters.
    /// Initializes the layer's parameters and allocates memory for weights, biases,
    /// and gradients. Throws an exception if any parameter is invalid.
    ///
    /// @param inputChannels The number of input channels (depth).
    /// @param numFilters The number of filters (output channels) in the layer.
    /// @param kernelSize The size of the kernel (filter).
    /// @param stride The stride of the convolution operation.
    /// @throws std::invalid_argument If any of the parameters are less than or equal to zero
    ConvolutionalLayer(int inputChannels, int numFilters, int kernelSize, int stride)
        : inputChannels_(inputChannels) // Set the number of input channels
        , numFilters_(numFilters)       // Set the number of filters (output channels)
        , kernelSize_(kernelSize)       // Set the kernel size
        , stride_(stride)               // Set the stride
    {

        // Validate input parameters to ensure they are greater than zero
        if (inputChannels <= 0 || numFilters <= 0 || kernelSize <= 0 || stride <= 0)
        {
            throw std::invalid_argument("Invalid layer parameters");
        }

        srand(static_cast<unsigned int>(time(nullptr))); // Seed for random number generation
        InitializeGradients();                           // Initialize gradients for weights and biases
        InitializeWeights();                             // Initialize weights with small random values
        InitializeBiases();                              // Initialize the biases with small random values
    }

    /// @brief Performs the forward pass of the convolutional layer.
    /// Takes the input data and computes the output of the convolution operation.
    ///
    /// @param input A 3D vector representing the input data (channels, height, width).
    /// @return A 3D vector representing the output of the convolution.
    std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        input_ = input; // Store input for use in backpropagation

        // Ensure the input has the correct number of channels
        if (input.size() != inputChannels_)
        {
            throw std::invalid_argument("Input channel size does not match the expected input channels.");
        }

        // Calculate output dimensions based on input size, kernel size, and stride
        int outputHeight = (input[0].size() - kernelSize_) / stride_ + 1;
        int outputWidth  = (input[0][0].size() - kernelSize_) / stride_ + 1;

        // Initialize output tensor with zeros for each filter
        std::vector<std::vector<std::vector<float>>> output(
            numFilters_, std::vector<std::vector<float>>(outputHeight, std::vector<float>(outputWidth, 0.0f)));

        try
        {
            // Loop over each filter to apply convolution
            for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
            {
                // Loop over output height for the current filter
                for (int outputHeightIndex = 0; outputHeightIndex < outputHeight;
                     ++outputHeightIndex) // Loop over output height
                {
                    // Loop over output width for the current filter
                    for (int outputWidthIndex = 0; outputWidthIndex < outputWidth;
                         ++outputWidthIndex) // Loop over output width
                    {
                        // Loop over each input channel
                        for (int channelIndex = 0; channelIndex < inputChannels_;
                             ++channelIndex) // Loop over input channels
                        {
                            // Loop over kernel height
                            for (int kernelHeightIndex = 0; kernelHeightIndex < kernelSize_;
                                 ++kernelHeightIndex) // Loop over kernel height
                            {
                                // Loop over kernel width
                                for (int kernelWidthIndex = 0; kernelWidthIndex < kernelSize_;
                                     ++kernelWidthIndex) // Loop over kernel width
                                {
                                    // Calculate the corresponding input index for height
                                    int inputHeightIndex = outputHeightIndex * stride_ + kernelHeightIndex;

                                    // Calculate the corresponding input index for width
                                    int inputWidthIndex = outputWidthIndex * stride_ + kernelWidthIndex;

                                    // Check input bounds to avoid accessing out-of-bounds memory
                                    if (inputHeightIndex < 0 || inputHeightIndex >= input[0].size() ||
                                        inputWidthIndex < 0 || inputWidthIndex >= input[0][0].size())
                                    {
                                        throw std::out_of_range("Input index out of bounds.");
                                    }

                                    // Accumulate the convolution results using the corresponding weight
                                    output[filterIndex][outputHeightIndex][outputWidthIndex] +=
                                        input[channelIndex][inputHeightIndex][inputWidthIndex] *
                                        weights_[filterIndex][channelIndex][kernelHeightIndex][kernelWidthIndex];
                                }
                            }
                        }
                        // Add bias to the convolution result for the current filter
                        output[filterIndex][outputHeightIndex][outputWidthIndex] += biases_[filterIndex][0];
                    }
                }
            }
        }
        catch (const std::exception& ex)
        {
            std::cerr << "An exception occurred in the ConvolutionalLayer class's Forward function: " << ex.what()
                      << std::endl;
        }

        return output; // Return the computed output of the convolution
    }

    /// @brief Performs the backward pass of the convolutional layer.
    /// This function calculates the gradients of the loss with respect to the inputs,
    /// weights, and biases using the output gradients from the next layer.
    /// It applies the learning rate to the weight gradients during backpropagation.
    ///
    /// @param outputGradient A 3D vector representing the gradient of the loss with respect to the output.
    /// @param learningRate The learning rate used for scaling the weight gradients.
    /// @return A 3D vector representing the gradient of the loss with respect to the input.
    std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& outputGradient, float learningRate) override
    {
        // Reset gradients for weights and biases to zero before calculating new gradients
        for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
        {
            for (int channelIndex = 0; channelIndex < inputChannels_; ++channelIndex) // Loop over each input channel
            {
                for (int kernelHeightIndex = 0; kernelHeightIndex < kernelSize_;
                     ++kernelHeightIndex) // Loop over the height of the kernel
                {
                    for (int kernelWidthIndex = 0; kernelWidthIndex < kernelSize_;
                         ++kernelWidthIndex) // Loop over the width of the kernel
                    {
                        // Reset the weight gradient for the current filter and kernel
                        weightGradients_[filterIndex][channelIndex][kernelHeightIndex][kernelWidthIndex] = 0.0f;
                    }
                }
            }
            // Reset the bias gradient for the current filter
            biasesGradient_[filterIndex][0] = 0.0f;
        }

        // Initialize input gradient tensor to store the gradients of the input data
        std::vector<std::vector<std::vector<float>>> inputGradient(
            inputChannels_,
            std::vector<std::vector<float>>(input_[0].size(), std::vector<float>(input_[0][0].size(), 0.0f)));

        // Calculate gradients for each filter and corresponding inputs
        for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
        {
            for (int outputHeightIndex = 0; outputHeightIndex < outputGradient[0].size();
                 ++outputHeightIndex) // Loop over the height of the output gradient
            {
                for (int outputWidthIndex = 0; outputWidthIndex < outputGradient[0][0].size();
                     ++outputWidthIndex) // Loop over the width of the output gradient
                {
                    // Loop over input channels to accumulate gradients
                    for (int channelIndex = 0; channelIndex < inputChannels_;
                         ++channelIndex) // Loop over each input channel
                    {
                        // Loop over the height of the kernel to calculate gradients
                        for (int kernelHeightIndex = 0; kernelHeightIndex < kernelSize_;
                             ++kernelHeightIndex) // Loop over the height of the kernel
                        {
                            for (int kernelWidthIndex = 0; kernelWidthIndex < kernelSize_;
                                 ++kernelWidthIndex) // Loop over the width of the kernel
                            {
                                // Compute corresponding input indices based on the stride
                                int inputHeightIndex = outputHeightIndex * stride_ + kernelHeightIndex;
                                int inputWidthIndex  = outputWidthIndex * stride_ + kernelWidthIndex;

                                // Accumulate input gradients and weight gradients if indices are within bounds
                                if (inputHeightIndex < input_[channelIndex].size() &&
                                    inputWidthIndex < input_[channelIndex][0].size())
                                {
                                    // Update the input gradient based on the output gradient and filter weights
                                    inputGradient[channelIndex][inputHeightIndex][inputWidthIndex] +=
                                        outputGradient[filterIndex][outputHeightIndex][outputWidthIndex] *
                                        weights_[filterIndex][channelIndex][kernelHeightIndex][kernelWidthIndex];

                                    // Update the weight gradient based on the input data and output gradient
                                    weightGradients_[filterIndex][channelIndex][kernelHeightIndex][kernelWidthIndex] +=
                                        learningRate *
                                        outputGradient[filterIndex][outputHeightIndex][outputWidthIndex] *
                                        input_[channelIndex][inputHeightIndex][inputWidthIndex];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Accumulate bias gradients for each filter based on the output gradient
        for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
        {
            for (int outputHeightIndex = 0; outputHeightIndex < outputGradient[0].size();
                 ++outputHeightIndex) // Loop over the height of the output gradient
            {
                for (int outputWidthIndex = 0; outputWidthIndex < outputGradient[0][0].size();
                     ++outputWidthIndex) // Loop over the width of the output gradient
                {
                    // Update the bias gradient by accumulating the output gradients
                    biasesGradient_[filterIndex][0] +=
                        outputGradient[filterIndex][outputHeightIndex][outputWidthIndex]; // Update bias gradient
                }
            }
        }

        return inputGradient; // Return the computed gradient of the input
    }

    /// @brief Updates the weights and biases using the computed gradients.
    /// This function applies a learning rate to the computed gradients to
    /// update the weights and biases.
    ///
    /// @param learningRate The learning rate used for weight updates.
    void UpdateWeights(float learningRate) override
    {
        for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
        {
            for (int channelIndex = 0; channelIndex < inputChannels_; ++channelIndex) // Loop over input channels
            {
                for (int kernelHeightIndex = 0; kernelHeightIndex < kernelSize_;
                     ++kernelHeightIndex) // Loop over kernel height
                {
                    for (int kernelWidthIndex = 0; kernelWidthIndex < kernelSize_;
                         ++kernelWidthIndex) // Loop over kernel width
                    {
                        // Update weights using the computed gradients
                        weights_[filterIndex][channelIndex][kernelHeightIndex][kernelWidthIndex] -=
                            learningRate *
                            weightGradients_[filterIndex][channelIndex][kernelHeightIndex][kernelWidthIndex];
                    }
                }
            }
            // Update biases using the computed gradient
            biases_[filterIndex][0] -= learningRate * biasesGradient_[filterIndex][0];
        }
    }

    /// @brief Displays the weights and biases of the convolutional layer.
    /// This function prints the weights and biases to the standard output for
    /// debugging purposes.
    void DisplayWeightsAndBiases() const
    {
        std::cout << "Weights:" << std::endl;                               // Print weights header
        for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
        {
            std::cout << "Filter " << filterIndex << ":" << std::endl;                // Print filter index
            for (int channelIndex = 0; channelIndex < inputChannels_; ++channelIndex) // Loop over input channels
            {
                for (int kernelHeightIndex = 0; kernelHeightIndex < kernelSize_;
                     ++kernelHeightIndex) // Loop over kernel height
                {
                    for (int kernelWidthIndex = 0; kernelWidthIndex < kernelSize_;
                         ++kernelWidthIndex) // Loop over kernel width
                    {
                        std::cout << weights_[filterIndex][channelIndex][kernelHeightIndex][kernelWidthIndex]
                                  << " "; // Print weight value
                    }
                    std::cout << std::endl; // New line after each kernel row
                }
            }
        }

        std::cout << "Biases:" << std::endl;                                // Print biases header
        for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
        {
            std::cout << biases_[filterIndex][0] << std::endl; // Print bias value
        }
    }

    /// @brief Retrieves all weights and biases of the convolutional layer.
    /// @return A vector containing all the weights and biases of the layer.
    std::vector<float> GetAllWeights() const override
    {
        std::vector<float> allWeights;

        // Flatten the 4D weights_ tensor into the allWeights vector
        for (const auto& filter : weights_) // Iterate over each filter
        {
            for (const auto& channel : filter) // Iterate over each input channel
            {
                for (const auto& row : channel) // Iterate over each row in the filter
                {
                    allWeights.insert(allWeights.end(), row.begin(), row.end()); // Add the row to allWeights
                }
            }
        }

        // Add biases to allWeights
        for (const auto& row : biases_) // Iterate over each row of the biases weights matrix
        {
            allWeights.insert(allWeights.end(), row.begin(), row.end()); // Flatten and append weights
        }

        return allWeights;
    }

    /// @brief Computes the L2 regularization loss for the layer.
    /// @param [in] l2RegularizationFactor The factor for scaling the regularization loss.
    /// @return The computed L2 regularization loss for this layer.
    float ComputeL2Regularization(float l2RegularizationFactor) const
    {
        return 0;
    }

  private:
    int inputChannels_; ///< Number of input channels (depth) for the convolution layer
    int numFilters_;    ///< Number of filters (output channels) in the convolution layer
    int kernelSize_;    ///< Size of the kernel (filter) used in the convolution
    int stride_;        ///< Stride length for the convolution operation
    std::vector<std::vector<std::vector<float>>> input_; ///< 3D vector storing input data (channels, height, width)
    std::vector<std::vector<std::vector<std::vector<float>>>> weights_; ///< 4D vector storing weights for each filter
    std::vector<std::vector<float>>                           biases_;  ///< 2D vector storing biases for each filter
    std::vector<std::vector<std::vector<std::vector<float>>>>
                                    weightGradients_; ///< 4D vector storing gradients for weights
    std::vector<std::vector<float>> biasesGradient_;  ///< 2D vector storing gradients for biases

    /// @brief Initializes weights with small random values.
    /// This method allocates and initializes the weights tensor with small random
    /// values to break symmetry.
    void InitializeWeights()
    {
        // Resize weights vector to accommodate filters, input channels, and kernel size
        weights_.resize(
            numFilters_,
            std::vector<std::vector<std::vector<float>>>(
                inputChannels_, std::vector<std::vector<float>>(kernelSize_, std::vector<float>(kernelSize_, 0.0f))));

        // Populate weights with small random values
        for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
        {
            for (int channelIndex = 0; channelIndex < inputChannels_; ++channelIndex) // Loop over input channels
            {
                for (int kernelHeightIndex = 0; kernelHeightIndex < kernelSize_;
                     ++kernelHeightIndex) // Loop over kernel height
                {
                    for (int kernelWidthIndex = 0; kernelWidthIndex < kernelSize_;
                         ++kernelWidthIndex) // Loop over kernel width
                    {
                        weights_[filterIndex][channelIndex][kernelHeightIndex][kernelWidthIndex] =
                            static_cast<float>(rand()) / RAND_MAX * 0.01f; // Assign small random values to weights
                    }
                }
            }
        }
    }

    /// @brief Initializes biases with small random values.
    /// This method allocates and initializes the biases for each filter with
    /// small random values.
    void InitializeBiases()
    {
        biases_.resize(numFilters_, std::vector<float>(1, 0.0f));           // Allocate biases for each filter
        for (int filterIndex = 0; filterIndex < numFilters_; ++filterIndex) // Loop over each filter
        {
            biases_[filterIndex][0] = static_cast<float>(rand()) / RAND_MAX * 0.01f; // Assign small random bias
        }
    }

    /// @brief Initializes gradients for weights and biases.
    /// This method allocates and initializes the gradients for weights and biases
    /// to zero, preparing them for gradient accumulation during backpropagation.
    void InitializeGradients()
    {
        // Resize weightGradients vector to accommodate filters, input channels, and kernel size
        weightGradients_.resize(
            numFilters_,
            std::vector<std::vector<std::vector<float>>>(
                inputChannels_, std::vector<std::vector<float>>(kernelSize_, std::vector<float>(kernelSize_, 0.0f))));

        // Allocate biases gradient for each filter
        biasesGradient_.resize(numFilters_, std::vector<float>(1, 0.0f)); // One bias gradient per filter
    }
};

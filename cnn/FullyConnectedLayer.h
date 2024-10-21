#pragma once

#include <random>
#include <vector>

#include "Layer.h" // Include the Layer class header

/// @class FullyConnectedLayer
/// @brief A class representing a fully connected layer in a neural network.
///
/// This class implements the forward pass, backward pass, and weight updates for a fully connected layer,
/// which connects all input neurons to all output neurons. It handles the weight and bias initialization
/// and the calculation of gradients during backpropagation.
class FullyConnectedLayer : public Layer // Inherit from Layer
{
  public:
    /// @brief Constructor for the FullyConnectedLayer class.
    /// Initializes the fully connected layer with the specified input size and output size.
    ///
    /// @param inputSize The size of the input vector.
    /// @param outputSize The size of the output vector.
    FullyConnectedLayer(int inputSize, int outputSize)
        : inputSize_(inputSize)
        , outputSize_(outputSize)
    {
        weights_.resize(outputSize_, std::vector<float>(inputSize_));
        biases_.resize(outputSize_, 0.0f);
        InitializeWeights(); // Call to initialize weights with random values
    }

    /// @brief Performs the forward pass of the fully connected layer.
    /// This function takes the input tensor, computes the output by performing
    /// matrix multiplication with weights and adds the biases.
    ///
    /// @param input A 3D vector representing the input tensor (channels, height, width).
    /// @return A 3D vector representing the output tensor after the forward pass.
    std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) override
    {
        // Flatten input for the fully connected operation
        std::vector<float> flattenedInput = FlattenInput(input); // Flatten the input tensor
        std::vector<float> output(outputSize_, 0.0f);            // Initialize output vector

        // Compute output by performing matrix multiplication and adding biases
        for (int i = 0; i < outputSize_; ++i)
        {
            output[i] = biases_[i]; // Start with bias for each output neuron
            for (int j = 0; j < inputSize_; ++j)
            {
                output[i] += weights_[i][j] * flattenedInput[j]; // Dot product
            }
        }

        // Store the flattened input for backward pass
        previousInput_ = flattenedInput;

        // Reshape output back to 3D tensor
        return ReshapeOutput(output); // Reshape output to 3D tensor format
    }

    /// @brief Performs the backward pass of the fully connected layer.
    /// This function computes the gradients of the loss with respect to the input tensor,
    /// weights, and biases based on the upstream gradient and the learning rate.
    ///
    /// @param upstreamGradient A 3D vector representing the gradient from the upstream layer (channels, height, width).
    /// @param learningRate The learning rate used for adjusting the gradients.
    /// @return A 3D vector representing the gradient with respect to the input tensor.
    std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& upstreamGradient, float learningRate) override
    {
        // Flatten upstream gradient for the fully connected layer
        std::vector<float> flattenedGradient = FlattenInput(upstreamGradient);      // Flatten the upstream gradient
        std::vector<float> inputGradient(inputSize_, 0.0f);                         // Initialize input gradient vector
        weightsGradient_.resize(outputSize_, std::vector<float>(inputSize_, 0.0f)); // Resize weights gradient
        biasesGradient_.resize(outputSize_, 0.0f);                                  // Resize biases gradient

        // Calculate gradients for weights and biases
        for (int i = 0; i < outputSize_; ++i) // Loop over each output neuron
        {
            biasesGradient_[i] = flattenedGradient[i]; // Gradient w.r.t. bias

            for (int j = 0; j < inputSize_; ++j) // Loop over each input neuron
            {
                weightsGradient_[i][j] = flattenedGradient[i] * previousInput_[j] *
                                         learningRate; // Gradient w.r.t. weights adjusted by learning rate
                inputGradient[j] += flattenedGradient[i] * weights_[i][j]; // Accumulate gradient w.r.t. input
            }
        }

        // Return gradient reshaped back to the 3D tensor format
        return ReshapeOutput(inputGradient); // Reshape input gradient if needed
    }

    /// @brief Updates the weights and biases of the fully connected layer.
    /// This function applies the computed gradients to update the weights and biases
    /// using the specified learning rate.
    ///
    /// @param learningRate The learning rate used for weight updates.
    void UpdateWeights(float learningRate) override
    {
        // Update weights using gradient descent
        for (int i = 0; i < outputSize_; ++i) // Loop over each output neuron
        {
            for (int j = 0; j < inputSize_; ++j) // Loop over each input neuron
            {
                weights_[i][j] -= learningRate * weightsGradient_[i][j]; // Update each weight
            }
        }

        // Update biases using gradient descent
        for (int i = 0; i < outputSize_; ++i) // Loop over each output neuron
        {
            biases_[i] -= learningRate * biasesGradient_[i]; // Update each bias
        }
    }

    /// @brief Returns a flattened vector of all the weights in the fully connected layer.
    /// This function combines all weights into a single 1D vector.
    ///
    /// @return A 1D vector containing all the weights from the fully connected layer.
    std::vector<float> GetAllWeights() const override
    {
        std::vector<float> allWeights;   // Vector to store all weights
        for (const auto& row : weights_) // Iterate over each row of the weight matrix
        {
            allWeights.insert(allWeights.end(), row.begin(), row.end()); // Flatten and append weights
        }
        return allWeights; // Return the 1D vector of weights
    }

    /// @brief Computes the L2 regularization loss for the layer.
    /// @param [in] l2RegularizationFactor The factor for scaling the regularization loss.
    /// @return The computed L2 regularization loss for this layer.
    float ComputeL2Regularization(float l2RegularizationFactor) const
    {
        return 0;
    }

  private:
    int                             inputSize_;       // The size of the input vector
    int                             outputSize_;      // The size of the output vector
    std::vector<std::vector<float>> weights_;         // The weight matrix connecting input to output
    std::vector<float>              biases_;          // The bias vector for the output neurons
    std::vector<float>              previousInput_;   // To store the input for backward propagation
    std::vector<std::vector<float>> weightsGradient_; // Gradients for weights
    std::vector<float>              biasesGradient_;  // Gradients for biases

    /// @brief Initializes weights with random values.
    /// This function uses a uniform distribution to initialize weights
    /// in the range of [-0.01, 0.01].
    void InitializeWeights()
    {
        std::default_random_engine            generator;                   // Random number generator
        std::uniform_real_distribution<float> distribution(-0.01f, 0.01f); // Distribution for weights

        // Populate weights with random values
        for (int i = 0; i < outputSize_; ++i) // Loop over each output neuron
        {
            for (int j = 0; j < inputSize_; ++j) // Loop over each input neuron
            {
                weights_[i][j] = distribution(generator); // Assign random weight
            }
        }
    }

    /// @brief Flattens a 3D input tensor into a 1D vector.
    /// This function concatenates all the values in the input tensor into a single vector.
    ///
    /// @param input A 3D vector representing the input tensor (channels, height, width).
    /// @return A 1D vector containing all values from the input tensor.
    std::vector<float> FlattenInput(const std::vector<std::vector<std::vector<float>>>& input)
    {
        std::vector<float> flattened;     // Vector to hold flattened input
        for (const auto& channel : input) // Iterate over channels
        {
            for (const auto& row : channel) // Iterate over rows
            {
                flattened.insert(flattened.end(), row.begin(), row.end()); // Concatenate row values
            }
        }
        return flattened; // Return the flattened vector
    }

    /// @brief Reshapes a 1D output vector back to a 3D tensor format.
    /// This function converts the output vector from the forward pass
    /// back into a 3D tensor format based on the specified dimensions.
    ///
    /// @param output A 1D vector representing the output from the forward pass.
    /// @return A 3D vector representing the reshaped output tensor.
    std::vector<std::vector<std::vector<float>>> ReshapeOutput(const std::vector<float>& output)
    {
        // Assuming output is returned as a 3D tensor for a specific shape.
        // Modify as needed based on the actual required output shape.
        // Here, I am returning as a single-channel output for demonstration.
        int                                          height = 1;             // Example height
        int                                          width  = output.size(); // Example width equals output size
        std::vector<std::vector<std::vector<float>>> reshapedOutput(
            1, std::vector<std::vector<float>>(height, std::vector<float>(width)));

        for (size_t i = 0; i < output.size(); ++i)
        {
            reshapedOutput[0][0][i] = output[i]; // Place output in first channel
        }

        return reshapedOutput; // Return the reshaped output tensor
    }
};

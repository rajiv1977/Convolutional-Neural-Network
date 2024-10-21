#pragma once

#include <vector>

/// @class Layer
/// @brief An abstract base class representing a layer in a neural network.
/// This class serves as a foundation for all types of neural network layers (e.g., convolutional layers,
/// pooling layers, etc.). It defines pure virtual functions that must be implemented by derived classes.
/// Specifically, it provides methods for the forward pass, backward pass, weight updates, and weight retrieval.
class Layer
{
  public:
    /// @brief Virtual destructor for the Layer class.
    /// Ensures that derived classes can properly handle their cleanup when objects of the derived type
    /// are deleted via a pointer to the base class (Layer). This is important for dynamic memory management.
    virtual ~Layer() = default;

    /// @brief Pure virtual function for performing the forward pass of the layer.
    /// This function must be implemented by any class that derives from `Layer`. It processes the input data
    /// and produces an output, typically transforming the data based on the type of layer (e.g., convolutional,
    /// pooling).
    ///
    /// @param [in] input A 3D tensor (vector of vectors) representing the input data to the layer, with dimensions
    /// (channels, height, width).
    /// @return A 3D tensor representing the output data after passing through the layer, with dimensions depending on
    /// the type of layer and its parameters.
    virtual std::vector<std::vector<std::vector<float>>>
        Forward(const std::vector<std::vector<std::vector<float>>>& input) = 0;

    /// @brief Pure virtual function for performing the backward pass of the layer.
    /// This function must be implemented by derived classes to compute gradients for the layer's weights
    /// and biases based on the output gradient received from the subsequent layer during backpropagation.
    ///
    /// @param [in] outputGradient A 3D tensor representing the gradient of the loss with respect to the output of this
    /// layer.
    /// @param [in] learningRate The learning rate for adjusting the weights during the update process.
    /// @return A 3D tensor representing the gradient of the loss with respect to the input of this layer, which will be
    /// passed to the previous layer.
    virtual std::vector<std::vector<std::vector<float>>>
        Backward(const std::vector<std::vector<std::vector<float>>>& outputGradient, float learningRate) = 0;

    /// @brief Pure virtual function for updating the weights of the layer.
    /// This function must be implemented by derived classes to apply the calculated gradients to the layer's
    /// weights and biases using a specified learning rate.
    ///
    /// @param [in] learningRate The learning rate for adjusting the weights during the update process.
    virtual void UpdateWeights(float learningRate) = 0;

    /// @brief Pure virtual function to retrieve all weights and biases of the layer.
    /// This function must be implemented by derived classes to return a vector containing all weights
    /// and biases associated with the layer.
    ///
    /// @return A vector containing all weights and biases of the layer.
    virtual std::vector<float> GetAllWeights() const = 0;

    /// @brief Computes the L2 regularization loss for the layer.
    /// @param [in] l2RegularizationFactor The factor for scaling the regularization loss.
    /// @return The computed L2 regularization loss for this layer.
    virtual float ComputeL2Regularization(float l2RegularizationFactor) const = 0;

  protected:
    std::vector<float> weights_; // Placeholder for weights; each layer will manage its own weights
    std::vector<float> biases_;  // Placeholder for biases; each layer will manage its own biases
};

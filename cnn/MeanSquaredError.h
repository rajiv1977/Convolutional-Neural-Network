#include "Loss.h"
#include <vector>
#include <cmath>

/// @brief Class that implements Mean Squared Error (MSE) loss function.
class MeanSquaredError : public Loss
{
  public:
    /// @brief Computes the mean squared error loss between true labels and predictions.
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted values.
    /// @return The computed mean squared error loss.
    float ComputeLoss(const std::vector<float>& trueLabels, const std::vector<float>& predictions) const override
    {
        float loss = 0.0f;
        // Iterate over predictions to calculate the squared differences
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            float diff = predictions[i] - trueLabels[i]; // Difference between prediction and true label
            loss += diff * diff;                         // Accumulate squared difference
        }
        return loss / predictions.size(); // Return mean loss
    }

    /// @brief Computes the gradient of the mean squared error loss.
    /// @param predictions A vector of predicted values.
    /// @param trueLabels A vector of true labels.
    /// @return A vector representing the gradients of the MSE loss.
    std::vector<float> ComputeGradient(const std::vector<float>& predictions,
                                       const std::vector<float>& trueLabels) const override
    {
        std::vector<float> gradients(predictions.size());
        // Calculate the gradient for each prediction
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            gradients[i] = (predictions[i] - trueLabels[i]) / predictions.size(); // Gradient of MSE
        }
        return gradients; // Return the computed gradients
    }

    /// @brief Computes focal loss between true labels and predictions (not implemented).
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted values.
    /// @param alpha Balancing factor for class imbalance.
    /// @param gamma Focusing parameter for adjusting down-weighting of easy examples.
    /// @return A placeholder value (0.0) since implementation is not provided.
    float ComputeFocalLoss(const std::vector<float>& trueLabels,
                           const std::vector<float>& predictions,
                           float                     alpha = 1.0f,
                           float                     gamma = 2.0f) const override
    {
        // Implement focal loss computation if needed
        return 0.0f; // Placeholder return value
    }

    /// @brief Computes the gradient of the focal loss (not implemented).
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted values.
    /// @param alpha Balancing factor for class imbalance.
    /// @param gamma Focusing parameter for adjusting down-weighting of easy examples.
    /// @return A placeholder empty vector since implementation is not provided.
    std::vector<float> ComputeFocalLossGradient(const std::vector<float>& trueLabels,
                                                const std::vector<float>& predictions,
                                                float                     alpha = 1.0f,
                                                float                     gamma = 2.0f) const override
    {
        // Implement focal loss gradient computation if needed
        return {}; // Placeholder return value
    }

    /// @brief Computes the gradient of the loss with respect to predictions.
    /// @param predictions A vector of predicted values.
    /// @param trueLabels A vector of true labels.
    /// @return A vector representing the gradients of the MSE loss.
    std::vector<float> ComputeLossGradient(const std::vector<float>& predictions,
                                           const std::vector<float>& trueLabels) override
    {
        std::vector<float> gradients(predictions.size());
        // Calculate the gradient for each prediction
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            gradients[i] = (predictions[i] - trueLabels[i]) / predictions.size(); // Gradient of MSE
        }
        return gradients; // Return the computed gradients
    }

    /// @brief Computes the L2 loss based on model weights.
    /// @param weights A vector of model weights.
    /// @param l2Regularization The strength of L2 regularization.
    /// @return The computed L2 loss.
    float ComputeL2Loss(const std::vector<float>& weights, float l2Regularization) const override
    {
        float l2Loss = 0.0f;
        // Iterate over each weight to calculate the sum of squares
        for (const float weight : weights)
        {
            l2Loss += weight * weight; // Accumulate squared weight
        }
        return l2Regularization * l2Loss; // Scale by the regularization strength
    }
};

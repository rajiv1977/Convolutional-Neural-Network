#pragma once

#include "Loss.h"
#include <vector>
#include <cmath>

class L2Loss : public Loss
{
  public:
    // Computes the L2 loss between true labels and predictions
    float ComputeLoss(const std::vector<float>& trueLabels, const std::vector<float>& predictions) const override
    {
        float loss = 0.0f;
        for (size_t i = 0; i < trueLabels.size(); ++i)
        {
            float diff = predictions[i] - trueLabels[i];
            loss += diff * diff; // Sum of squared differences
        }
        return loss / trueLabels.size(); // Mean L2 loss
    }

    // Computes the gradient of the L2 loss
    std::vector<float> ComputeGradient(const std::vector<float>& predictions,
                                       const std::vector<float>& trueLabels) const override
    {
        std::vector<float> gradient(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            gradient[i] = 2 * (predictions[i] - trueLabels[i]); // Derivative of L2 loss
        }
        return gradient;
    }

    // Computes the focal loss between true labels and predictions
    float ComputeFocalLoss(const std::vector<float>& trueLabels,
                           const std::vector<float>& predictions,
                           float                     alpha = 1.0f,
                           float                     gamma = 2.0f) const override
    {
        // Optionally implement focal loss for L2
        return ComputeLoss(trueLabels, predictions); // Can reuse existing loss calculation
    }

    // Computes the gradient of the focal loss
    std::vector<float> ComputeFocalLossGradient(const std::vector<float>& trueLabels,
                                                const std::vector<float>& predictions,
                                                float                     alpha = 1.0f,
                                                float                     gamma = 2.0f) const override
    {
        // For L2 loss, we might choose to return the gradient from ComputeGradient
        return ComputeGradient(predictions, trueLabels); // Reuse gradient calculation
    }

    // Computes the loss gradient using double precision
    std::vector<float> ComputeLossGradient(const std::vector<float>& predictions,
                                            const std::vector<float>& trueLabels) override
    {
        std::vector<float> gradient(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            gradient[i] = 2 * (predictions[i] - trueLabels[i]); // Derivative of L2 loss
        }
        return gradient;
    }

    /// @brief Computes the L2 loss based on weights.
    /// @param weights A vector of model weights.
    /// @param l2Regularization The strength of L2 regularization.
    /// @return The computed L2 loss.
    float ComputeL2Loss(const std::vector<float>& weights, float l2Regularization) const override
    {
        float l2Loss = 0.0f;
        for (const float weight : weights)
        {
            l2Loss += weight * weight; // Squared weights
        }
        return l2Regularization * l2Loss; // Scale by the regularization strength
    }
};

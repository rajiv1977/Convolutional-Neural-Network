#pragma once

#include "Loss.h"
#include <cmath>
#include <vector>

/// @brief Class that implements Focal Loss for addressing class imbalance in classification tasks.
class FocalLoss : public Loss
{
  public:
    /// @brief Constructor to initialize alpha and gamma for focal loss.
    /// @param alpha Balancing factor to address class imbalance.
    /// @param gamma Focusing parameter to adjust the rate at which easy examples are down-weighted.
    FocalLoss(float alpha = 1.0f, float gamma = 2.0f)
        : alpha_(alpha)
        , gamma_(gamma)
    {
    }

    /// @brief Computes the focal loss between true labels and predicted probabilities.
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted probabilities.
    /// @return The computed focal loss.
    float ComputeLoss(const std::vector<float>& trueLabels, const std::vector<float>& predictions) const override
    {
        float loss = 0.0f;
        // Iterate over each true label and prediction
        for (size_t i = 0; i < trueLabels.size(); ++i)
        {
            // Calculate the predicted probability for the true class
            float p_t = trueLabels[i] * predictions[i] + (1 - trueLabels[i]) * (1 - predictions[i]);
            // Accumulate the focal loss for the current prediction
            loss -= alpha_ * std::pow((1 - p_t), gamma_) * std::log(p_t);
        }
        return loss; // Return the total focal loss
    }

    /// @brief Computes the gradient of the focal loss with respect to predictions.
    /// @param predictions A vector of predicted probabilities.
    /// @param trueLabels A vector of true labels.
    /// @return A vector representing the gradient of the focal loss.
    std::vector<float> ComputeGradient(const std::vector<float>& predictions,
                                       const std::vector<float>& trueLabels) const override
    {
        std::vector<float> gradient(predictions.size());
        // Iterate over each prediction to calculate the gradient
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            float p_t = trueLabels[i] * predictions[i] + (1 - trueLabels[i]) * (1 - predictions[i]);
            // Calculate the focal gradient for the current prediction
            float focal_grad =
                -alpha_ * std::pow((1 - p_t), gamma_) * (gamma_ * std::log(p_t) - (p_t > 0 ? 1.0f / p_t : 0.0f));
            gradient[i] = focal_grad; // Store the computed gradient
        }
        return gradient; // Return the vector of gradients
    }

    /// @brief Computes focal loss based on true labels and predictions.
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted probabilities.
    /// @param alpha Balancing factor for class imbalance.
    /// @param gamma Focusing parameter for adjusting down-weighting of easy examples.
    /// @return The computed focal loss.
    float ComputeFocalLoss(const std::vector<float>& trueLabels,
                           const std::vector<float>& predictions,
                           float                     alpha = 1.0f,
                           float                     gamma = 2.0f) const override
    {
        return ComputeLoss(trueLabels, predictions); // Reuse existing loss calculation
    }

    /// @brief Computes the gradient of the focal loss.
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted probabilities.
    /// @param alpha Balancing factor for class imbalance.
    /// @param gamma Focusing parameter for adjusting down-weighting of easy examples.
    /// @return A vector representing the gradient of the focal loss.
    std::vector<float> ComputeFocalLossGradient(const std::vector<float>& trueLabels,
                                                const std::vector<float>& predictions,
                                                float                     alpha = 1.0f,
                                                float                     gamma = 2.0f) const override
    {
        std::vector<float> gradient(predictions.size());
        // Iterate over each prediction to calculate the gradient
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            float p_t = trueLabels[i] * predictions[i] + (1 - trueLabels[i]) * (1 - predictions[i]);
            // Calculate the focal gradient for the current prediction
            float focal_grad =
                -alpha * std::pow((1 - p_t), gamma) * (gamma * std::log(p_t) - (p_t > 0 ? 1.0f / p_t : 0.0f));
            gradient[i] = focal_grad; // Store the computed gradient
        }
        return gradient; // Return the vector of gradients
    }

    /// @brief Computes the gradient of the loss with respect to predictions.
    /// @param predictions A vector of predicted probabilities.
    /// @param trueLabels A vector of true labels.
    /// @return A vector representing the loss gradient.
    std::vector<float> ComputeLossGradient(const std::vector<float>& predictions,
                                           const std::vector<float>& trueLabels) override
    {
        std::vector<float> gradient(predictions.size());
        // Iterate over each prediction to calculate the loss gradient
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            float p_t = trueLabels[i] * predictions[i] + (1 - trueLabels[i]) * (1 - predictions[i]);
            // Calculate the focal gradient for the current prediction
            float focal_grad =
                -alpha_ * std::pow((1 - p_t), gamma_) * (gamma_ * std::log(p_t) - (p_t > 0 ? 1.0 / p_t : 0.0));
            gradient[i] = focal_grad; // Store the computed gradient
        }
        return gradient; // Return the vector of gradients
    }

    /// @brief Computes the L2 loss based on the model weights.
    /// @param weights A vector of model weights.
    /// @param l2Regularization The strength of L2 regularization.
    /// @return The computed L2 loss.
    float ComputeL2Loss(const std::vector<float>& weights, float l2Regularization) const override
    {
        float l2Loss = 0.0f;
        // Iterate over each weight to calculate the sum of squares
        for (const float weight : weights)
        {
            l2Loss += weight * weight; // Accumulate the squared weight
        }
        return l2Regularization * l2Loss; // Scale by the regularization strength
    }

  private:
    float alpha_; // Balancing factor for class imbalance
    float gamma_; // Focusing parameter for down-weighting easy examples
};

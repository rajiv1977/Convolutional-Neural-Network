#ifndef LOSS_H
#define LOSS_H

#include <vector>

class Loss
{
  public:
    virtual ~Loss() = default;

    /// @brief Computes the loss between true labels and predictions.
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted outputs.
    /// @return The computed loss.
    virtual float ComputeLoss(const std::vector<float>& trueLabels, const std::vector<float>& predictions) const = 0;

    /// @brief Computes the gradient of the loss.
    /// @param predictions A vector of predicted outputs.
    /// @param trueLabels A vector of true labels.
    /// @return A vector representing the gradient of the loss.
    virtual std::vector<float> ComputeGradient(const std::vector<float>& predictions,
                                               const std::vector<float>& trueLabels) const = 0;

    /// @brief Computes the focal loss between true labels and predictions.
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted outputs (probabilities).
    /// @param alpha The balancing factor for class imbalance (default: 1.0).
    /// @param gamma The focusing parameter (default: 2.0).
    /// @return The computed focal loss.
    virtual float ComputeFocalLoss(const std::vector<float>& trueLabels,
                                   const std::vector<float>& predictions,
                                   float                     alpha = 1.0f,
                                   float                     gamma = 2.0f) const = 0; // Pure virtual function

    /// @brief Computes the gradient of the focal loss.
    /// @param trueLabels A vector of true labels.
    /// @param predictions A vector of predicted outputs (probabilities).
    /// @param alpha The balancing factor for class imbalance (default: 1.0).
    /// @param gamma The focusing parameter (default: 2.0).
    /// @return A vector representing the gradient of the focal loss.
    virtual std::vector<float> ComputeFocalLossGradient(const std::vector<float>& trueLabels,
                                                        const std::vector<float>& predictions,
                                                        float                     alpha = 1.0f,
                                                        float gamma = 2.0f) const = 0; // Pure virtual function

    /// @brief Virtual method to compute the loss gradient; should be implemented in derived classes.
    /// @param predictions A vector of predicted outputs.
    /// @param trueLabels A vector of true labels.
    /// @return A vector representing the loss gradient.
    virtual std::vector<float> ComputeLossGradient(const std::vector<float>& predictions,
                                                    const std::vector<float>& trueLabels) = 0; // Pure virtual function

    /// @brief Computes the L2 loss contribution based on the model weights.
    /// @param weights A vector of model weights.
    /// @param l2Regularization The strength of L2 regularization.
    /// @return The computed L2 loss.
    virtual float ComputeL2Loss(const std::vector<float>& weights, float l2Regularization) const = 0;
};

#endif // LOSS_H

#include <cmath>

#include "Loss.h"

class CrossEntropyLoss : public Loss
{
  public:
    float ComputeLoss(const std::vector<float>& trueLabels, const std::vector<float>& predictions) const override
    {
        float loss = 0.0f;
        for (size_t i = 0; i < trueLabels.size(); ++i)
        {
            loss -= trueLabels[i] * std::log(predictions[i] + 1e-10); // Adding a small value to avoid log(0)
        }
        return loss;
    }

    std::vector<float> ComputeGradient(const std::vector<float>& predictions,
                                       const std::vector<float>& trueLabels) const override
    {
        std::vector<float> gradients(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            gradients[i] = predictions[i] - trueLabels[i]; // Gradient for cross-entropy loss
        }
        return gradients;
    }

    float ComputeFocalLoss(const std::vector<float>& trueLabels,
                           const std::vector<float>& predictions,
                           float                     alpha = 1.0f,
                           float                     gamma = 2.0f) const override
    {
        float loss = 0.0f;
        for (size_t i = 0; i < trueLabels.size(); ++i)
        {
            loss -= alpha * std::pow(1 - predictions[i], gamma) * trueLabels[i] * std::log(predictions[i] + 1e-10);
        }
        return loss;
    }

    std::vector<float> ComputeFocalLossGradient(const std::vector<float>& trueLabels,
                                                const std::vector<float>& predictions,
                                                float                     alpha = 1.0f,
                                                float                     gamma = 2.0f) const override
    {
        std::vector<float> gradients(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i)
        {
            float p_t    = predictions[i];
            gradients[i] = alpha * std::pow(1 - p_t, gamma) * (trueLabels[i] - p_t); // Gradient for focal loss
        }
        return gradients;
    }

    std::vector<float> ComputeLossGradient(const std::vector<float>& predictions,
                                           const std::vector<float>& trueLabels) override
    {
        return ComputeGradient(predictions, trueLabels); // Use the same gradient method for standard loss
    }
};

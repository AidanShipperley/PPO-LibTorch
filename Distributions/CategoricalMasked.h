#ifndef CATEGORICAL_MASKED_H
#define CATEGORICAL_MASKED_H

#include <iostream>
#include <torch/torch.h>
#include <math.h>
#include <vector>

class CategoricalMasked {

public:
    CategoricalMasked();
    CategoricalMasked(const torch::Tensor& logits, const torch::Tensor& masks, std::shared_ptr<torch::Device> device);
    ~CategoricalMasked();

    torch::Tensor logits_to_probs(torch::Tensor logits, bool is_binary = false);
    torch::Tensor sample();
    torch::Tensor log_prob(torch::Tensor value);
    torch::Tensor entropy();
    torch::Tensor mean();
    torch::Tensor mode();
    torch::Tensor variance();
    torch::Tensor enumerate_support();

    torch::Tensor m_logits, m_probs, m_masks;
    int64_t m_num_events;

    std::shared_ptr<torch::Device> m_device;

};

#endif
#ifndef CATEGORICAL_H
#define CATEGORICAL_H

#include <iostream>
#include <torch/torch.h>
#include <math.h>

class Categorical {

public:
    Categorical();
    Categorical(const torch::Tensor& logits, std::shared_ptr<torch::Device> device);
    ~Categorical();

    torch::Tensor logits_to_probs(torch::Tensor logits, bool is_binary = false);
    torch::Tensor sample();
    torch::Tensor log_prob(torch::Tensor value);
    torch::Tensor entropy();
    torch::Tensor mean();
    torch::Tensor mode();
    torch::Tensor variance();
    torch::Tensor enumerate_support();

    torch::Tensor m_logits, m_probs;
    int64_t m_num_events;

    std::shared_ptr<torch::Device> m_device;

};

#endif
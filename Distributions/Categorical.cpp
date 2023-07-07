/***********************************************************
 ** File:    Categorical.cpp
 ** Project: PPO
 ** Author:  Aidan Shipperley
 ** Date     8/25/2022
 ** This file contains an implementation of the categorical
 ** distribution that's not included in the C++ version of
 ** PyTorch. This only contains the necessary functions
 ** to be implemented with PPO and a few extra
 **********************************************************/

#include "Categorical.h"

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CategoricalMasked() -> Default Constructor
// -------------------------
// Should not be used
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Categorical::Categorical() {
    m_num_events = 0;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CategoricalMasked() -> Overloaded Constructor
// -------------------------
// Creates a categorical distribution from logits
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Categorical::Categorical(const torch::Tensor& logits, std::shared_ptr<torch::Device> device) {

    // This is just how logits are initialized, so I'm keeping everything identical
    // (https://pytorch.org/docs/stable/_modules/torch/distributions/categorical.html#Categorical)
    // Normalize
    m_logits = logits - logits.logsumexp(/*dim=*/-1, /*keepdim=*/true);
    m_probs = logits_to_probs(logits);
    m_num_events = m_probs.sizes().back();

    m_device = device;

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~CategoricalMasked() -> Destructor
// -------------------------
// Deallocates/destroys anything we used
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Categorical::~Categorical() {

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// logits_to_probs() -> returns torch::Tensor
// -------------------------
// Converts a tensor of logits into probabilities. 
// 
// Note that for the binary case, each value denotes 
// log odds, whereas for the multi - dimensional case, 
// the values along the last dimension denote the log 
// probabilities(possibly unnormalized) of the events.
// 
// Converted from:
// https://github.com/pytorch/pytorch/blob/c29502dd2fa38c79ada620fbde2f61d58df6e219/torch/distributions/utils.py#L67-L76
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor Categorical::logits_to_probs(torch::Tensor logits, bool is_binary /*= false*/) {

    if (is_binary) {
        return torch::sigmoid(logits);
    }
    return torch::nn::functional::softmax(logits, torch::nn::functional::SoftmaxFuncOptions(-1));
}


// ALL BELOW CONVERTED FROM: (https://pytorch.org/docs/stable/_modules/torch/distributions/categorical.html#Categorical)
torch::Tensor Categorical::sample() {
    //torch::NoGradGuard no_grad;

    torch::Tensor probs_2d = m_probs.reshape({ -1, m_num_events });
    torch::Tensor samples_2d = torch::multinomial(probs_2d, 1, true).t();
    return samples_2d.reshape(m_probs.sizes()[0]); // TEST THIS

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// log_prob() -> returns torch::Tensor
// -------------------------
// Returns the logarithm of the probabilities given,
// which just represents the probabilities on a 
// logarithmic scale instead of the standard
// [0, 1] interval.
// 
// Converted from:
// https://pytorch.org/docs/stable/_modules/torch/distributions/categorical.html#Categorical.log_prob
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor Categorical::log_prob(torch::Tensor value) {

    value = value.to(torch::kLong).unsqueeze(-1);
    std::vector<torch::Tensor> valueAndLog_pmf = torch::broadcast_tensors({ value, m_logits });
    value = valueAndLog_pmf[0];
    torch::Tensor log_pmf = valueAndLog_pmf[1];
    value = value.index({ "...", torch::indexing::Slice(torch::indexing::None, 1) });
    return log_pmf.gather(-1, value).squeeze(-1);

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// entropy() -> returns torch::Tensor
// -------------------------
// Returns the entropy of the categorical
// distribution
// 
// Converted from:
// https://pytorch.org/docs/stable/_modules/torch/distributions/categorical.html#Categorical.entropy
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor Categorical::entropy() {

    float min_real = std::numeric_limits<float>::min();
    torch::Tensor logits = torch::clamp(m_logits, min_real);
    torch::Tensor p_log_p = logits * m_probs;
    return -p_log_p.sum(-1);

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// mean() -> returns torch::Tensor
// -------------------------
// Returns mean of the categorical distribution
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor Categorical::mean() {
    return torch::full({ m_probs.sizes()[0] }, std::numeric_limits<float>::quiet_NaN(), torch::TensorOptions().device(*m_device).dtype(m_probs.dtype()));
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// mode() -> returns torch::Tensor
// -------------------------
// Returns mode of the categorical distribution
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor Categorical::mode() {
    return torch::argmax(m_probs, -1);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// variance() -> returns torch::Tensor
// -------------------------
// Returns variance of the categorical distribution
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor Categorical::variance() {
    return torch::full({ m_probs.sizes()[0] }, std::numeric_limits<float>::quiet_NaN(), torch::TensorOptions().device(*m_device).dtype(m_probs.dtype()));
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// enumerate_support() -> returns torch::Tensor
// -------------------------
// Returns tensor with all possible values in the 
// support of the categorical distribution
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor Categorical::enumerate_support() {
    torch::Tensor values = torch::arange(m_num_events, torch::TensorOptions().dtype(torch::kLong).device(*m_device));
    values = values.view({ -1, 1 }).expand({ -1, m_probs.sizes()[0] });
    return values;
}
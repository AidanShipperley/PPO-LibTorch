#ifndef AGENT_H
#define AGENT_H

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <random>
#include <math.h>
#include <numeric>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include <tuple>
#include <exception>

#include "Distributions/Categorical.h"
#include "Distributions/CategoricalMasked.h"


// Define agent return structs for readability of code
struct AgentOutput {
    torch::Tensor action;
    torch::Tensor logprob;
    torch::Tensor entropy;
    torch::Tensor value;
};


class Agent : public torch::nn::Module {

public:
    Agent(int64_t obsSize, int64_t actionSize, std::shared_ptr<torch::Device> device);
    ~Agent();
    torch::nn::Linear ppoLayerInit(torch::nn::Linear layer, const double stdDev = sqrt(2), const double bias_const = 0.0);
    torch::Tensor getValue(const torch::Tensor& x);
    AgentOutput getActionAndValueDiscrete(const torch::Tensor& x, torch::Tensor action = torch::Tensor());
    AgentOutput getActionAndValueMasked(const torch::Tensor& x, const torch::Tensor& mask, torch::Tensor action = torch::Tensor());
    void printAgent();

    torch::nn::Sequential           m_Critic;
    torch::nn::Sequential           m_Actor;
    std::shared_ptr<torch::Tensor>  m_actor_logstd;

    // Identical to MultiDiscrete action space, the size of this
    // vector must be identical to the number of actions you want
    // your agent to take
    std::vector<int64_t>            m_actionSpace;
    int64_t                         m_actionSpaceSum;
    int64_t                         m_actionSpaceSize;
    //std::ofstream m_logfile;
    std::shared_ptr<torch::Device>  m_device;

};







#endif
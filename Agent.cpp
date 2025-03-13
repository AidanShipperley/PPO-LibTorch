/**********************************************************
 ** File:    Agent.cpp
 ** Project: RocketPPO
 ** Author:  Aidan Shipperley
 ** Date     3/1/2022
 ** This file contains the agent to be used in PPO,
 ** it's initialization, and functions to get the
 ** actor and critic's inference.
 **********************************************************/

#include "Agent.h"

 //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 // Agent() -> Overloaded Constructor
 // -------------------------
 // Creates PPO's agent, which contains the actor
 // and the critic's networks.
 //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Agent::Agent(int64_t obsSize, int64_t actionSize, std::shared_ptr<torch::Device> device) {

    m_actionSpace = { actionSize };//{ 3, 3, 3, 3, 3, 2, 2, 2 };
    m_actionSpaceSum = std::accumulate(m_actionSpace.begin(), m_actionSpace.end(), static_cast<int64_t>(0));

    // Initialize all linear layers here
    torch::nn::Linear criticInputLayer = ppoLayerInit(torch::nn::Linear(obsSize, 64));
    torch::nn::Linear criticMiddleLayer = ppoLayerInit(torch::nn::Linear(64, 64));
    //torch::nn::Linear criticMiddleLayer2 = ppoLayerInit(torch::nn::Linear(256, 256));
    torch::nn::Linear criticOutputLayer = ppoLayerInit(torch::nn::Linear(64, 1), 1.0); // Output layer uses 1 as std deviation

    torch::nn::Linear actorInputLayer = ppoLayerInit(torch::nn::Linear(obsSize, 64));
    torch::nn::Linear actorMiddleLayer = ppoLayerInit(torch::nn::Linear(64, 64));
    //torch::nn::Linear actorMiddleLayer2 = ppoLayerInit(torch::nn::Linear(256, 256));
    
    // Output layer uses 0.01 as std deviation.
    // This ensures layer's parameters will have similar scalar values
    // so that the probability of taking each action will be similar
    torch::nn::Linear actorOutputLayer = ppoLayerInit(torch::nn::Linear(64, m_actionSpaceSum), 0.01);

    // Critic's network is composed of 3 linear layers w/ hyperbolic tan as activation function
    torch::nn::Sequential critic({
        {"criticInputLayer" , criticInputLayer      },
        {"Tanh1"            , torch::nn::Tanh()     },
        {"criticMiddleLayer", criticMiddleLayer     },
        {"Tanh2"            , torch::nn::Tanh()     },
        //{"criticMiddleLayer2", criticMiddleLayer2   },
        //{"Tanh3"            , torch::nn::Tanh()     },
        {"criticOutputLayer", criticOutputLayer     }
        });

    // Actor's network is very similar to critic except for it's output linear layer being initialized with different stdDev
    torch::nn::Sequential action_mean({
        {"actorInputLayer" , actorInputLayer    },
        {"Tanh1"           , torch::nn::Tanh()  },
        {"actorMiddleLayer", actorMiddleLayer   },
        {"Tanh2"           , torch::nn::Tanh()  },
        //{"actorMiddleLayer2", actorMiddleLayer2 },
        //{"Tanh3"           , torch::nn::Tanh()  },
        {"actorOutputLayer", actorOutputLayer   }
        });

    m_Critic = critic;
    m_Actor = action_mean;
    //m_actor_logstd = std::make_shared<torch::Tensor>(at::zeros({ 1, actionSize }));

    m_Critic = register_module("m_Critic", m_Critic);
    m_Actor = register_module("m_Actor", m_Actor);
    // We don't generate std directly, rather we generate the log(std).
    // These are just learnable nn parameters that don't take any inputs, making them state independent
    //register_parameter("m_actor_logstd", *m_actor_logstd);
    m_device = device;

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~Agent() -> Destructor
// -------------------------
// Deallocates/destroys anything we used
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Agent::~Agent() {

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ppoLayerInit() -> returns torch::nn::Linear
// -------------------------
// Funciton to properly initialize all of the layers
// in the actor and critic networks. PPO uses 
// orthogonal initialization on the layer's weight 
// and the constant initialization on the layer's bias
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::nn::Linear Agent::ppoLayerInit(torch::nn::Linear layer, const double stdDev, const double bias_const) {

    torch::NoGradGuard noGrad;
    torch::nn::init::orthogonal_(layer->weight, stdDev);
    torch::nn::init::constant_(layer->bias, bias_const);

    return layer;

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// getValue() -> returns torch::Tensor
// -------------------------
// Implement critic's inference by passing obs to 
// the critic's network
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor Agent::getValue(const torch::Tensor& x) {
    return m_Critic->forward(x);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// getActionAndValueDiscrete() -> returns std::array<torch::Tensor, 4>
// -------------------------
// Implement actors inference bundled with critic's 
// inference.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AgentOutput Agent::getActionAndValueDiscrete(const torch::Tensor& x, torch::Tensor action) {

    torch::Tensor logits = m_Actor->forward(x);
    Categorical categorical = Categorical(logits, m_device);

    if (!action.numel()) {
        action = categorical.sample();
    }

    return { action, categorical.log_prob(action.squeeze()), categorical.entropy(), m_Critic->forward(x) };

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// getActionAndValueMasked() -> returns std::vector<torch::Tensor>
// -------------------------
// Implement actors inference bundled with critic's 
// inference. This also contains the logic to handle
// invalid action masking.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AgentOutput Agent::getActionAndValueMasked(const torch::Tensor& x, const torch::Tensor& mask, torch::Tensor action) {

    torch::Tensor logits = m_Actor->forward(x);
    std::vector<torch::Tensor> split_logits = torch::split(logits, m_actionSpace, 1);
    std::vector<torch::Tensor> split_action_masks = torch::split(mask, m_actionSpace, 1);

    std::vector<CategoricalMasked> multi_categoricals(split_logits.size());

    std::vector<torch::Tensor> multi_categorical_samples(multi_categoricals.size());
    std::vector<torch::Tensor> multi_categorical_logprob(multi_categoricals.size());
    std::vector<torch::Tensor> multi_categorical_entropy(multi_categoricals.size());

    for (int64_t i = 0; i < split_logits.size(); i++) {
        multi_categoricals[i] = CategoricalMasked(split_logits[i], split_action_masks[i], m_device);
        if (!action.numel()) {
            multi_categorical_samples[i] = multi_categoricals[i].sample();
        }
    }

    if (!action.numel()) {
        action = torch::stack(multi_categorical_samples);
    }

    for (int64_t i = 0; i < multi_categoricals.size(); i++) {
        multi_categorical_logprob[i] = multi_categoricals[i].log_prob(action[i]);
        multi_categorical_entropy[i] = multi_categoricals[i].entropy();
    }

    torch::Tensor logprob = torch::stack(multi_categorical_logprob);
    torch::Tensor entropy = torch::stack(multi_categorical_entropy);

    return { action.t(), logprob.sum(0), entropy.sum(0), m_Critic->forward(x) };

}


void Agent::printAgent() {

    for (auto& p : named_parameters()) {
        std::cout << p.key() << std::endl;
        std::cout << p.value() << std::endl;
    }

    std::cout << "Agent's m_Critic: \n" << m_Critic << std::endl;
    std::cout << "Agent's m_Actor: \n" << m_Actor << std::endl;
    std::cout << "Agent's m_actor_logstd: \n" << *m_actor_logstd << std::endl;

}
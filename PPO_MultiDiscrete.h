#ifndef PPO_MULTI_DISCRETE_H
#define PPO_MULTI_DISCRETE_H

#include <iostream>
#include <filesystem> // create dirs to save models, search for args
#include <string>
#include <algorithm> // std::fill
#include <execution> // std::execution::par
#include <chrono> // Everything to keep track of timing
#include <array> // Returning tensors in functions cleanly
#include <numeric> // std::accumulate
#include <deque> // episode lengths and rewards logging
#include <torch/torch.h>

#include "Agent.h"
#include "ThreadPool.h"
#include "Utils.h"
#include "Environments/MountainCar.h"

class PPO_MultiDiscrete {
public:

    PPO_MultiDiscrete();
    ~PPO_MultiDiscrete();

    // Setup
    void getArgs();
    void loadPolicyFromCheckpoint();

    // ALGO LOGIC 
    std::vector<torch::Tensor> computeActionLogic(const torch::Tensor& next_obs, \
        const torch::Tensor& action_mask, const torch::Tensor& action);
    std::array<torch::Tensor, 2> calcAdvantage(const torch::Tensor& next_obs, const torch::Tensor& next_done);
    torch::Tensor getApproxKLAndClippedObj(const torch::Tensor& ratio, const torch::Tensor& logratio);

    void train();

    // Controlling Environments
    torch::Tensor initEnvs(const torch::Tensor& action_mask);
    std::array<torch::Tensor, 3> stepEnvs(const torch::Tensor& action);

    // Printing results to console
    void printPPOResults(int64_t update, int64_t global_step, std::chrono::milliseconds fps, std::chrono::milliseconds time_elapsed,
                         torch::Tensor& approx_kl, torch::Tensor& entropy_loss, torch::Tensor& explained_var, torch::Tensor& loss,
                         torch::Tensor& pg_loss, torch::Tensor& v_loss);
    template<typename T> void printElement(T t, const int& width);
    float getVectorMean(std::vector<float> vector);

    // Hyperparameters
    int64_t                                 m_obs_size;
    int64_t                                 m_action_size;
    float                                   m_action_high;
    float                                   m_action_low;
    float                                   m_learning_rate;          // The learning rate of the experiment
    int64_t                                 m_seed;                   // The seed of the experiment for consistent test results
    int64_t                                 m_total_timesteps;        // The total timesteps of the experiments
    bool                                    m_use_cuda;               // Whether or not to attempt to use cuda. Will default to cpu if no device is found
    bool                                    m_torch_deterministic;    // Whether or not to make all torch calculations deterministic/reproducible. 
                                                                      // Should probably always be true
    int64_t                                 m_num_envs;               // The number of parallel game environments to run
    int64_t                                 m_num_steps;              // The number of steps to run in each environment per policy rollout
    bool                                    m_anneal_lr;              // Whether or not to anneal the learning rate over time
    bool                                    m_use_gae;                // Whether or not to use Generalized Advantage Estimation
    float                                   m_gamma;                  // Gamma hyperparameter of GAE
    float                                   m_gae_lambda;             // Lambda hyperparameter of GAE
    int64_t                                 m_num_minibatches;        // Number of training minibatches per update 
    int64_t                                 m_update_epochs;          // Number of epochs when optimizing the surrogate loss
    bool                                    m_norm_adv;               // Whether or not to normalize the advantage
    float                                   m_clip_coef;              // Clipping parameter
    bool                                    m_clip_vloss;             // Whether or not to clip the value function loss
    float                                   m_ent_coef;               // Entropy coefficient for the loss calculation
    float                                   m_vf_coef;                // Value function coefficient for the loss calculation
    float                                   m_max_grad_norm;          // The maximum value for gradient clipping

    int64_t                                 m_checkpoint_updates;     // How often you would like to checkpoint your model and optimizer
    int64_t                                 m_max_episode_steps;      // How long an episode can run before automatically terminating

    // Calculated from other variables
    int64_t                                 m_batch_size;
    int64_t                                 m_minibatch_size;
    std::shared_ptr<torch::Device>          m_device;

    // PPO Modules
    std::shared_ptr<Agent>                  m_agent;
    //torch::optim::Adam                      m_optimizer(){};
    std::shared_ptr<torch::optim::AdamW>    m_optimizer;

    // Storage variables
    torch::Tensor                           m_obs = torch::Tensor(); // Must specify default constructor for tensors       
    torch::Tensor                           m_actions = torch::Tensor();
    torch::Tensor                           m_logprobs = torch::Tensor();
    torch::Tensor                           m_rewards = torch::Tensor();
    torch::Tensor                           m_dones = torch::Tensor();
    torch::Tensor                           m_values = torch::Tensor();
    torch::Tensor                           m_action_masks = torch::Tensor();

    // Logging variables
    //std::shared_ptr<TensorBoardLogger>    m_logger;
    std::vector<float>                      m_clipfracs;
    std::deque<int64_t>                     m_episode_lengths;
    std::deque<float>                       m_episode_rewards;
    uint64_t                                m_global_step;

    // Thread pool
    std::shared_ptr<ThreadPool>             m_threadPool;

    // Environment Variables
    std::vector<std::shared_ptr<MountainCar>> m_envs;

};

template<typename T>
inline void PPO_MultiDiscrete::printElement(T t, const int& width)
{
    std::cout << std::left << std::setw(width) << std::setfill(' ') << t << std::right << "|\n";
}

#endif
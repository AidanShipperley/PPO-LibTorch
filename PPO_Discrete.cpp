
#include "PPO_Discrete.h"

PPO_Discrete::PPO_Discrete() {

    // First initialize all PPO hyperparameters to their defaults
    m_obs_size = 2;
    m_action_size = 1;
    m_learning_rate = 0.0003;
    m_seed = 1;
    m_total_timesteps = 100000;
    m_use_cuda = true;
    m_torch_deterministic = true;
    m_num_envs = 1;
    m_num_steps = 2048;
    m_anneal_lr = false;
    m_use_gae = true;
    m_gamma = 0.99f;
    m_gae_lambda = 0.95f;
    m_num_minibatches = 64;
    m_update_epochs = 10;
    m_norm_adv = true;
    m_clip_coef = 0.2f;
    m_clip_vloss = true;
    m_ent_coef = 0.01f;
    m_vf_coef = 0.5f;
    m_max_grad_norm = 0.5f;
    m_checkpoint_updates = 5;
    m_max_episode_steps = 500;

    m_batch_size = int(m_num_envs * m_num_steps);
    m_minibatch_size = int(m_batch_size / m_num_minibatches);

    m_global_step = 0;

    // Now check if there's a config file, and if so, update hyperparams based on sheet
    PPO_Discrete::getArgs();

    // Init threadpool
    m_threadPool = std::make_shared<ThreadPool>(std::thread::hardware_concurrency());

    // Seed randomization
    srand(m_seed);
    torch::manual_seed(m_seed);
    at::globalContext().setDeterministicCuDNN(m_torch_deterministic ? true : false);
    // https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    at::globalContext().setDeterministicAlgorithms(m_torch_deterministic ? true : false, true);

    // Global Speedups
    // Enable optimized cuDNN algorithms, works best with non-fluxuating input size, perfect for RL
    // https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    at::globalContext().setBenchmarkCuDNN(true);

    // Use float32 tensor cores on Ampere GPUs, less precision for ~7x speedup
    // https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);

    // Used FP16 mixed precision
    // https://pytorch.org/docs/stable/notes/cuda.html#reduced-precision-reduction-in-fp16-gemms
    at::globalContext().setAllowFP16ReductionCuBLAS(true);

    // Initialize device
    m_device = std::make_shared<torch::Device>((((torch::cuda::is_available()) && (m_use_cuda)) ? torch::kCUDA : torch::kCPU));
    std::cout << "Using " << *m_device << " device" << std::endl;

    // Initialize Agent
    std::cout << "m_obs_size: " << m_obs_size << std::endl;
    std::cout << "m_action_size: " << m_action_size << std::endl;

    m_agent = std::make_shared<Agent>(m_obs_size, m_action_size + 1, m_device);
    m_agent->to(*m_device);

    // Initialize Adam optimizer with respect to agents parameters
    m_optimizer = std::make_shared<torch::optim::AdamW>(
        m_agent->parameters(), torch::optim::AdamWOptions(m_learning_rate).eps(1e-5)
    );

    // If any checkpoints exist, load them and resume previous training
    loadPolicyFromCheckpoint();

    // Initialize Environments
    for (int i = 0; i < m_num_envs; i++) {
        m_envs.push_back(std::make_shared<CartPole>(m_seed));
    }
    std::cout << "made envs" << std::endl;

    // Initialize storage variables
    m_obs = torch::zeros({ m_num_steps, m_num_envs, m_obs_size }).to(*m_device);
    m_actions = torch::zeros({ m_num_steps, m_num_envs, m_action_size }).to(*m_device);
    m_logprobs = torch::zeros({ m_num_steps, m_num_envs }).to(*m_device);
    m_rewards = torch::zeros({ m_num_steps, m_num_envs }).to(*m_device);
    m_dones = torch::zeros({ m_num_steps, m_num_envs }).to(*m_device);
    m_values = torch::zeros({ m_num_steps, m_num_envs }).to(*m_device);

    // Init episode length/reward loggers

}

PPO_Discrete::~PPO_Discrete() {


}

void PPO_Discrete::getArgs() {

    std::vector<std::string> hyperParamStrings = {
            "obs_size","action_size","learning_rate","seed","total_timesteps","use_cuda","torch_deterministic","num_envs",\
                "num_steps","anneal_lr","use_gae","gamma","gae_lambda","num_minibatches","update_epochs","norm_adv","clip_coef",\
                    "clip_vloss","ent_coef","vf_coef","max_grad_norm", "checkpoint_updates", "max_episode_steps"
    };

    std::string configFilePath = "./PPOConfig.txt";

    // Warn user if config params not found
    if (!std::filesystem::exists(configFilePath)) {
        std::cout << "Could not find " << configFilePath.c_str() << " file." << \
            "\nUsing default PPO hyperparameters" << std::endl;
    }
    else {

        std::ifstream configFile(configFilePath);
        std::string configText = "";

        // Loop through whole file for specific hyperparameters
        while (getline(configFile, configText)) {
            configText = PPOUtils::formatString(configText); // Remove whitespaces and makes lowercase
            // Check to make sure hyperparameter is valid
            bool identified = false;
            for (int i = 0; i < hyperParamStrings.size(); i++) {
                if (hyperParamStrings.at(i) == configText.substr(0, configText.find("="))) {
                    identified = true;

                    if (hyperParamStrings.at(i) == "obs_size") {
                        m_obs_size = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file obs_size = " << m_obs_size << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "action_size") {
                        m_action_size = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file action_size = " << m_action_size << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "learning_rate") {
                        m_learning_rate = stof(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file learning_rate = " << m_learning_rate << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "seed") {
                        m_seed = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file seed = " << m_seed << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "total_timesteps") {
                        m_total_timesteps = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file total_timesteps = " << m_total_timesteps << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "use_cuda") {
                        m_use_cuda = (configText.substr(configText.find("=") + 1) == "true") ? true : false;
                        std::cout << "Using config file use_cuda = " << m_use_cuda << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "torch_deterministic") {
                        m_torch_deterministic = (configText.substr(configText.find("=") + 1) == "true") ? true : false;
                        std::cout << "Using config file torch_deterministic = " << m_torch_deterministic << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "num_envs") {
                        m_num_envs = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file num_envs = " << m_num_envs << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "num_steps") {
                        m_num_steps = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file num_steps = " << m_num_steps << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "anneal_lr") {
                        m_anneal_lr = (configText.substr(configText.find("=") + 1) == "true") ? true : false;
                        std::cout << "Using config file anneal_lr = " << m_anneal_lr << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "use_gae") {
                        m_use_gae = (configText.substr(configText.find("=") + 1) == "true") ? true : false;
                        std::cout << "Using config file use_gae = " << m_use_gae << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "m_gamma") {
                        m_gamma = stof(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file gamma = " << m_gamma << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "m_gae_lambda") {
                        m_gae_lambda = stof(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file gae_lambda = " << m_gae_lambda << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "num_minibatches") {
                        m_num_minibatches = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file num_minibatches = " << m_num_minibatches << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "update_epochs") {
                        m_update_epochs = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file update_epochs = " << m_update_epochs << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "norm_adv") {
                        m_norm_adv = (configText.substr(configText.find("=") + 1) == "true") ? true : false;
                        std::cout << "Using config file norm_adv = " << m_norm_adv << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "clip_coef") {
                        m_clip_coef = stof(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file clip_coef = " << m_clip_coef << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "clip_vloss") {
                        m_clip_vloss = (configText.substr(configText.find("=") + 1) == "true") ? true : false;
                        std::cout << "Using config file clip_vloss = " << m_clip_vloss << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "ent_coef") {
                        m_ent_coef = stof(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file ent_coef = " << m_ent_coef << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "vf_coef") {
                        m_vf_coef = stof(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file vf_coef = " << m_vf_coef << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "max_grad_norm") {
                        m_max_grad_norm = stof(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file max_grad_norm = " << m_max_grad_norm << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "checkpoint_updates") {
                        m_checkpoint_updates = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file checkpoint_updates = " << m_checkpoint_updates << std::endl;
                    }

                    else if (hyperParamStrings.at(i) == "max_episode_steps") {
                        m_max_episode_steps = stoi(configText.substr(configText.find("=") + 1));
                        std::cout << "Using config file max_episode_steps = " << m_max_episode_steps << std::endl;
                    }

                }
            }
            if (!identified) {
                std::cout << "Unknown hyperparameter in \"PPOConfig.txt\" file: \"" << configText.substr(0, configText.find("=")) \
                    << "\"" << std::endl;
                exit(0);
            }
        }

        m_batch_size = int(m_num_envs * m_num_steps);
        m_minibatch_size = int(m_batch_size / m_num_minibatches);
    }

}

// Action logic (no_grad scope)
std::vector<torch::Tensor> PPO_Discrete::computeActionLogic(const torch::Tensor& next_obs, const torch::Tensor& input_action)
{

    torch::NoGradGuard no_grad;

    std::vector<torch::Tensor> actionsAndValues = m_agent->getActionAndValueDiscrete(next_obs, input_action);

    return { actionsAndValues[0],
            actionsAndValues[1],
            actionsAndValues[2],
            actionsAndValues[3],
            actionsAndValues[3].flatten() };

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// calcAdvantage() -> std::array<torch::Tensor, 2>
// -------------------------
// This function calculates the advantage, which is an estimate 
// of how much better an action is compared to the average action 
// at a given state.It uses either the Generalized Advantage 
// Estimation(GAE) algorithm or a regular advantage calculation, 
// depending on whether the m_use_gae variable is true or false.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
std::array<torch::Tensor, 2> PPO_Discrete::calcAdvantage(const torch::Tensor& next_obs, const torch::Tensor& next_done) {

    torch::NoGradGuard no_grad;

    torch::Tensor nextnonterminal = torch::Tensor();
    // bootstrap value if not done
    torch::Tensor next_value = m_agent->getValue(next_obs).reshape({ 1, -1 });

    // GAE implementation
    if (m_use_gae) {
        torch::Tensor nextvalues = torch::Tensor();

        torch::Tensor advantages = torch::zeros_like(m_rewards).to(*m_device);
        torch::Tensor lastgaelam = torch::zeros({ 1, m_num_envs }).to(*m_device);

        for (int t = m_num_steps - 1; t >= 0; t--) {

            if (t == m_num_steps - 1) {
                nextnonterminal = 1.0 - next_done;
                nextvalues = next_value;
            }
            else {
                nextnonterminal = 1.0 - m_dones[t + 1];
                nextvalues = m_values[t + 1];
            }

            torch::Tensor delta = m_rewards[t] + m_gamma * nextvalues * nextnonterminal - m_values[t];
            lastgaelam = delta + m_gamma * m_gae_lambda * nextnonterminal * lastgaelam;
            advantages[t] = lastgaelam[0];

        }
        return { advantages + m_values, advantages };
    }

    // Regular advantage calculation
    else {

        torch::Tensor next_return = torch::Tensor();
        
        torch::Tensor returns = torch::zeros_like(m_rewards).to(*m_device);
        for (int t = m_num_steps - 1; t >= 0; t--) {

            if (t == m_num_steps - 1) {
                nextnonterminal = 1.0 - next_done;
                next_return = next_value;
            }
            else {
                nextnonterminal = 1.0 - m_dones[t + 1];
                next_return = returns[t + 1];
            }
            returns[t] = m_rewards[t] + m_gamma * nextnonterminal * next_return;
        }

        return { returns, returns - m_values };

    }

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// getApproxKLAndClippedObj() -> torch::Tensor
// -------------------------
// This function computes the approximate Kullback - Leibler(KL) 
// divergence and the objective with clipped ratios. The KL divergence 
// is a measure of how one probability distribution differs from a 
// second, expected probability distribution. The clipped objective 
// is a way of limiting the updates to the policy during optimization 
// to make training more stable.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor PPO_Discrete::getApproxKLAndClippedObj(const torch::Tensor& ratio, const torch::Tensor& logratio) {

    torch::NoGradGuard no_grad;

    // Measure of how often the clipped objective is actually triggered
    torch::Tensor clippedRatios = torch::gt((ratio - 1.0).abs(), m_clip_coef);
    m_clipfracs.push_back(float(clippedRatios.count_nonzero().item<int>()) / float(clippedRatios.size(0)));

    //torch::Tensor old_approx_kl = (-logratio.mean()); // Original implementation approximates -log ratio
    torch::Tensor approx_kl = ((ratio - 1) - logratio).mean(); // Better estimator has been recently found

    return approx_kl;

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// initEnvs() -> torch::Tensor
// -------------------------
// This function initializes the environments by creating an 
// observation tensor for each environment and resetting each 
// one. The observations are stored in a tensor and returned.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch::Tensor PPO_Discrete::initEnvs() {

    // Construct initial_obs
    torch::Tensor obs = torch::zeros({ m_num_envs, m_obs_size }, torch::TensorOptions(*m_device).dtype(torch::kFloat32));

    // Put data into initial obs for each agent
    for (int i = 0; i < m_num_envs; i++) {

        // // Add job to pool to get initial observation from env
        m_threadPool->queueJob([this, i, &obs]() mutable {

            // Create empty tensor observation
            torch::Tensor tensor_obs = torch::zeros(m_obs_size, torch::kFloat32);

            // Reset the environment and get a float vector of the state
            std::vector<float> vec_obs = m_envs[i]->reset();

            // Copy float vector to tensor (Note: This is by far the fastest method to convert float vectors to tensors)
            std::memcpy(tensor_obs.data_ptr(), vec_obs.data(), sizeof(float) * tensor_obs.numel());
            obs[i] = tensor_obs;

        });

    }

    // Wait for all threads to finish
    m_threadPool->waitForJobsToFinish();

    return obs;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// stepEnvs() -> std::array<torch::Tensor, 3>
// -------------------------
// This function steps through each environment with the given action. 
// It creates new tensors for observations, rewards, and done statuses, 
// steps through the environment with the provided action, and checks 
// if the environment has terminated.If it has, it resets the 
// environment and logs the reward and length of the episode.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
std::array<torch::Tensor, 3> PPO_Discrete::stepEnvs(const torch::Tensor& action) {
    
    // Construct initial vars
    torch::Tensor obs = torch::zeros({ m_num_envs, m_obs_size }, torch::TensorOptions(*m_device).dtype(torch::kFloat32));
    torch::Tensor reward = torch::zeros({ m_num_envs, 1 }, torch::TensorOptions(*m_device).dtype(torch::kFloat32));
    torch::Tensor done = torch::zeros({ m_num_envs, 1 }, torch::TensorOptions(*m_device).dtype(torch::kInt32));

    // Convert action tensor to long long (int64_t) vector
    std::vector<int64_t> vec_action(action.data_ptr<int64_t>(), action.data_ptr<int64_t>() + action.numel());

    // Create vector to log done and rewards
    std::vector<bool> envs_finished(m_num_envs, false);
    std::vector<int> envs_length(m_num_envs, 0);
    std::vector<float> envs_reward(m_num_envs, 0);

    // Run a step in each environment
    for (int i = 0; i < m_num_envs; i++) {

        // // Add job to pool to get initial observation from env
        m_threadPool->queueJob([this, i, &obs, &reward, &done, &vec_action, &envs_finished, &envs_length, &envs_reward]() mutable {

            // Create empty tensor observation
            torch::Tensor tensor_obs = torch::zeros(m_obs_size, torch::kFloat32);
            torch::Tensor tensor_reward = torch::zeros(1, torch::kFloat32);
            torch::Tensor tensor_done = torch::zeros(1, torch::kInt32);

            // Step the environment
            auto [vec_obs, f_reward, b_terminated, b_info] = m_envs[i]->step(vec_action[i]);

            // Terminate the environment if we've reached the max episode steps
            if (m_envs[i]->episode_length == m_max_episode_steps) {
                b_terminated = true;
            }

            // If the i-th sub-environment is done (terminated or truncated) after stepping with the 
            // i-th action action[i], envs would set its returned next_done[i] to 1, auto-reset the 
            // i-th sub-environment and fill next_obs[i] with the initial observation in the new 
            // episode of the i-th environment.
            //
            // Reset i-th sub-env if done
            if (b_terminated) {
                envs_finished[i] = true;
                envs_length[i] = m_envs[i]->episode_length;
                envs_reward[i] = m_envs[i]->episode_reward;
                vec_obs = m_envs[i]->reset();
            }

            // Copy float vector to tensor (Note: This is by far the fastest method to convert float vectors to tensors)
            std::memcpy(tensor_obs.data_ptr(), vec_obs.data(), sizeof(float) * tensor_obs.numel());
            obs[i] = tensor_obs;
            reward[i] = f_reward;
            done[i] = b_terminated;

        });

    }

    // Wait for all threads to finish
    m_threadPool->waitForJobsToFinish();

    // Log episode reward and length on episode completion
    for (int i = 0; i < envs_finished.size(); i++) {

        if (envs_finished[i]) {

            while (m_episode_rewards.size() >= (10 * m_num_envs)) { 
                m_episode_rewards.pop_front();
                m_episode_lengths.pop_front();
            }
            m_episode_rewards.push_back(envs_reward[i]);
            m_episode_lengths.push_back(envs_length[i]);

        }

    }
    
    return { obs, reward, done };
}





void PPO_Discrete::train() {
    
    // Start the threads in a thread pool
    m_threadPool->start();

    // Initialize starting variables
    unsigned int global_step = m_global_step;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point update_time = std::chrono::steady_clock::now();
    torch::Tensor next_obs = torch::zeros({ m_num_envs, m_obs_size }, torch::kFloat32).to(*m_device);
    torch::Tensor next_done = torch::zeros({ m_num_envs }).to(*m_device);
    int num_updates = int((m_total_timesteps - global_step) / m_batch_size);

    // Setup environments
    next_obs = initEnvs();
    
    //    Variables required for scope in C++   //
    torch::Tensor v_loss = torch::empty(1).requires_grad_();
    torch::Tensor pg_loss = torch::empty(1).requires_grad_();
    torch::Tensor entropy_loss = torch::empty(1).requires_grad_();
    torch::Tensor loss = torch::empty(1).requires_grad_();
    torch::Tensor approx_kl = torch::Tensor();
    torch::Tensor explained_var = torch::Tensor();
    //////////////////////////////////////////////

    // Training loop
    for (int update = 1; update < num_updates + 1; update++) {

        // Annealing the learning rate if instructed to do so
        if (m_anneal_lr) {
            double frac = 1.0 - (update - 1.0) / num_updates;
            double lr_now = frac * m_learning_rate;
            static_cast<torch::optim::AdamWOptions&>(m_optimizer->param_groups()[0].options()).lr() = lr_now;
        }

        torch::Tensor reward = torch::Tensor();
        torch::Tensor done = torch::Tensor();

        // Policy Rollout Loop (where we run the environment with the current policy)
        for (int step = 0; step < m_num_steps; step++) {

            global_step += 1 * m_num_envs;

            // Start by assigning previous gathered next_obs and next_done
            m_obs[step] = next_obs;
            m_dones[step] = next_done;

            // During rollouts, we don't need to cache any gradients, so we compute
                    // actions, logprobs, and values under torch's no_grad context
            std::vector<torch::Tensor> result = computeActionLogic(next_obs, torch::Tensor());

            torch::Tensor action = result[0].unsqueeze(1);
            torch::Tensor logprob = result[1];
            torch::Tensor value = result[3];
            m_values[step] = result[4];
            m_actions[step] = action;
            m_logprobs[step] = logprob;

            // Step the environment(s) with the calculated action from agent
            std::array<torch::Tensor, 3> tensors = stepEnvs(action.cpu());
            next_obs = tensors[0];
            reward = tensors[1];
            done = tensors[2];

            // Gather obs, rewards, and dones into policy rollout buffer
            m_rewards[step] = reward.to(*m_device).view(-1);
            next_obs = next_obs.to(*m_device);
            next_done = done.squeeze().to(*m_device);

        }


        //////////////////////////    Learn with the batch of data collected    //////////////////////////

        // Calculate the advantage w/ or w/out GAE
        auto [returns, advantages] = calcAdvantage(next_obs, next_done);

        // Flatten the batch
        torch::Tensor b_obs = m_obs.reshape({ m_batch_size, m_obs_size });
        torch::Tensor b_logprobs = m_logprobs.reshape(-1);
        torch::Tensor b_actions = m_actions.reshape({ m_batch_size, m_action_size });
        torch::Tensor b_advantages = advantages.reshape(-1);
        torch::Tensor b_returns = returns.reshape(-1);
        torch::Tensor b_values = m_values.reshape(-1);

        std::vector<float>().swap(m_clipfracs); // Reset logging var

        // Optimizing the policy and value network
        for (int epoch = 0; epoch < m_update_epochs; epoch++) {

            torch::Tensor b_inds = torch::randperm(m_batch_size); // We create shuffled indicies so that each 
                                                                  // minibatch contains an equal # of randomized items

            // Break up m_batch_size batch into mini batches for training
            for (int start = 0; start < m_batch_size; start += m_minibatch_size) {

                int end = start + m_minibatch_size;
                torch::Tensor mb_inds = b_inds.index({ torch::indexing::Slice(start, end) }); // Create a slice of the tensors to index

                // Start with a forward pass on the minibatch observations, using the minibatched actions to keep the agent from sampling new actions
                std::vector<torch::Tensor> actionsAndValues = m_agent->getActionAndValueDiscrete(
                    b_obs.index({ mb_inds }),
                    b_actions.to(torch::kLong).index({ mb_inds })
                );

                // Logarithmic subtraction between the new log probabilities - old log probabilities associated w/ the actions in the policy rollout phase
                torch::Tensor logratio = actionsAndValues[1] - b_logprobs.index({ mb_inds });
                torch::Tensor ratio = logratio.exp();

                approx_kl = getApproxKLAndClippedObj(ratio, logratio);

                // Get advantages and normalize them if it should
                torch::Tensor mb_advantages = b_advantages.index({ mb_inds });
                if (m_norm_adv) {
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8); // small scalar value to prevent div 0 err
                }

                // Clipped objective -- Policy loss
                torch::Tensor pg_loss1 = -mb_advantages * ratio;
                torch::Tensor pg_loss2 = -mb_advantages * torch::clamp(ratio, 1 - m_clip_coef, 1 + m_clip_coef);
                pg_loss = torch::max(pg_loss1, pg_loss2).mean(); // policy loss does max of negatives, which is
                                                                 // equivalent to min of positives like in paper

                // Clipping value loss
                torch::Tensor newvalue = actionsAndValues[3].view(-1);
                if (m_clip_vloss) { // Original implementation uses clipping of value loss

                    torch::Tensor v_loss_unclipped = (newvalue - b_returns.index({ mb_inds })) * \
                        (newvalue - b_returns.index({ mb_inds }));

                    torch::Tensor v_clipped = b_values.index({ mb_inds }) + torch::clamp(
                        newvalue - b_values.index({ mb_inds }),
                        -m_clip_coef,
                        m_clip_coef
                    );

                    torch::Tensor v_loss_clipped = (v_clipped - b_returns.index({ mb_inds })) * \
                        (v_clipped - b_returns.index({ mb_inds }));

                    torch::Tensor v_loss_max = torch::max(v_loss_unclipped, v_loss_clipped);
                    v_loss = 0.5 * v_loss_max.mean();

                }
                else { // Normally, value loss is implemented as a mse between the predicted values and emperical returns
                    v_loss = 0.5 * ((newvalue - b_returns.index({ mb_inds })) * \
                        (newvalue - b_returns.index({ mb_inds }))).mean();
                }

                // Entropy loss -- measure of chaos in action probability distribution
                entropy_loss = actionsAndValues[2].mean();

                // Combine losses to get final loss value
                loss =                  (pg_loss)       -     (m_ent_coef * entropy_loss)     +       (v_loss * m_vf_coef);
                // Idea is to:    minimize policy loss			maximize entropy loss				and minimize value loss
                // Intuitively, maximizing entropy encourages the agent to explore more


                // Backprop
                m_optimizer->zero_grad();
                loss.backward();
                // Global gradient clipping specific to PPO
                torch::nn::utils::clip_grad_norm_(m_agent->parameters(), m_max_grad_norm);
                m_optimizer->step();

            }
        }

        // Get explained variance, tells you if your value function is a good indicator of the returns
        torch::Tensor var_y = b_returns.var();
        torch::Tensor explained_var = 1 - ((b_returns - b_values).var() / var_y);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time);
        auto fps = std::chrono::duration_cast<std::chrono::milliseconds>(end - update_time);

        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Log data to console
        printPPOResults(update, global_step, fps, time_elapsed, approx_kl, entropy_loss, explained_var, loss, pg_loss, v_loss);

        update_time = std::chrono::steady_clock::now();

        // Checkpoint save
        if (update % m_checkpoint_updates == 0) {
            std::filesystem::path modelSavePath = "./ModelCheckpoints/";
            std::filesystem::path optimizerSavePath = "./OptimizerCheckpoints/";
            std::filesystem::create_directories(modelSavePath);     // Create directories if they don't exist
            std::filesystem::create_directories(optimizerSavePath); // Create directories if they don't exist
            std::string agentFileName = modelSavePath.string() + "PPO_Agent_" + std::to_string(global_step) + "_steps.pt";
            std::string optimizerFileName = optimizerSavePath.string() + "PPO_Optimizer_" + std::to_string(global_step) + "_steps.pt";
            std::cout << "Saving model checkpoint to " << agentFileName << "..." << std::endl;
            torch::save(m_agent, agentFileName);
            std::cout << "Saving optimizer checkpoint to " << optimizerFileName << "..." << std::endl;
            torch::save(*m_optimizer, optimizerFileName);
        }

    }

    // Save final model
    std::filesystem::path modelSavePath = "./Models/";
    std::filesystem::create_directories(modelSavePath); // Create directories if they don't exist
    std::string agentFileName = modelSavePath.string() + "PPO_Agent_" + std::to_string(m_total_timesteps) + "_steps.pt";
    std::string optimizerFileName = modelSavePath.string() + "PPO_Optimizer_" + std::to_string(m_total_timesteps) + "_steps.pt";
    std::cout << "Saving model " << agentFileName << "..." << std::endl;
    torch::save(m_agent, agentFileName);
    std::cout << "Saving optimizer " << optimizerFileName << "..." << std::endl;
    torch::save(*m_optimizer, optimizerFileName);

    // Stops all threads
    m_threadPool->stop();

}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// printPPOResults() -> Returns void
// ---------------------------------
// Prints all relevant training data to the console
// window. This is meant to mimic and be practically
// identical to the printout generated by SB2/SB3
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void PPO_Discrete::printPPOResults(int update, int global_step, std::chrono::milliseconds fps, std::chrono::milliseconds time_elapsed,
                          torch::Tensor& approx_kl, torch::Tensor& entropy_loss, torch::Tensor& explained_var, torch::Tensor& loss,
                          torch::Tensor& pg_loss, torch::Tensor& v_loss) {

    // Log limited data for first update
    if (update == 1) {
        std::cout << "---------------------------------\n";
        if (m_episode_lengths.size() != 0) {
            std::cout << "| rollout/           |          |\n" <<
                std::setprecision(1) << std::defaultfloat <<
                "|    ep_len_mean     | ";
            printElement(std::accumulate(m_episode_lengths.begin(), m_episode_lengths.end(), 0) / m_episode_lengths.size(), 9);
            std::cout << std::setprecision(5) <<
                "|    ep_rew_mean     | ";
            printElement(std::accumulate(m_episode_rewards.begin(), m_episode_rewards.end(), 0.0f) / m_episode_rewards.size(), 9);
        }
        std::cout << "| time/              |          |\n" <<
            "|    fps             | ";
        printElement(int(m_batch_size / (fps.count() / 1000.0)), 9);
        std::cout << "|    iterations      | ";
        printElement(update, 9);
        std::cout << "|    time_elapsed    | ";
        printElement(int(time_elapsed.count() / 1000.0), 9);
        std::cout << "|    total_timesteps | ";
        printElement(global_step, 9);
        std::cout << "---------------------------------" << std::endl << std::endl;

    }

    // Log all data when not first update
    else {
        float currentLearningRate = float(static_cast<torch::optim::AdamWOptions&>(m_optimizer->param_groups()[0].options()).lr());

        std::cout << "------------------------------------------\n";
        if (m_episode_lengths.size() != 0) {
            std::cout << "| rollout/                |              |\n" <<
                std::setprecision(2) << std::fixed <<
                "|    ep_len_mean          | ";
            printElement(std::accumulate(m_episode_lengths.begin(), m_episode_lengths.end(), 0) / m_episode_lengths.size(), 13);
            std::cout << std::setprecision(8) <<
                "|    ep_rew_mean          | ";
            printElement(std::accumulate(m_episode_rewards.begin(), m_episode_rewards.end(), 0.0f) / m_episode_rewards.size(), 13);
        }
        std::cout << "| time/                   |              |\n" <<
            "|    fps                  | ";
        printElement(int(m_batch_size / (fps.count() / 1000.0)), 13);
        std::cout << "|    iterations           | ";
        printElement(update, 13);
        std::cout << "|    time_elapsed         | ";
        printElement(int(time_elapsed.count() / 1000.0), 13);
        std::cout << "|    total_timesteps      | ";
        printElement(global_step, 13);
        std::cout << "| train/                  |              |\n" <<
            std::setprecision(9) <<
            "|    approx_kl            | ";
        printElement(approx_kl.item<float>(), 13);
        std::cout << "|    clip_fraction        | ";
        printElement(PPOUtils::getVectorMean(m_clipfracs), 13);
        std::cout << "|    clip_range           | ";
        printElement(m_clip_coef, 13);
        std::cout << "|    entropy_loss         | " << std::fixed;
        printElement(entropy_loss.item<float>(), 13);
        std::cout << "|    explained_variance   | ";
        printElement(explained_var.item<float>(), 13);
        std::cout << "|    learning_rate        | " << std::defaultfloat << std::setprecision(6);
        printElement(currentLearningRate, 13);
        std::cout << "|    loss                 | " << std::fixed << std::setprecision(9);
        printElement(loss.item<float>(), 13);
        std::cout << "|    n_updates            | ";
        printElement(update * m_update_epochs, 13);
        std::cout << "|    policy_gradient_loss | ";
        printElement(pg_loss.item<float>(), 13);
        std::cout << "|    value_loss           | ";
        printElement(v_loss.item<float>(), 13);
        std::cout << "------------------------------------------" << std::endl << std::endl;

    }

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// loadPolicyFromCheckpoint() -> Returns void
// ---------------------------------
// Attempts to load a checkpointed model and
// optimizer if there is one in the bakkesmod data folder
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void PPO_Discrete::loadPolicyFromCheckpoint() {

    // Check if both dirs exist
    std::filesystem::path modelSavePath =  "./ModelCheckpoints/";
    std::filesystem::path optimSavePath = "./OptimizerCheckpoints/";

    if (!std::filesystem::exists(modelSavePath) || !std::filesystem::exists(optimSavePath)) {
        std::cout << "No previous model checkpoint found at " << modelSavePath << ", initializing new agent!" << std::endl;
        return;
    }

    // Load agent from file (actor/critic network, logstd parameter)
    if (std::distance(std::filesystem::directory_iterator(modelSavePath), std::filesystem::directory_iterator{}) == 0) {
        std::cout << "No previous model checkpoint found at " << modelSavePath << ", initializing new agent!" << std::endl;
    }
    else {
        std::string agentFileName;
        std::filesystem::file_time_type agentWriteTime;
        // directory_iterator can be iterated using a range-for loop
        for (auto const& dir_entry : std::filesystem::directory_iterator{ modelSavePath })
        {
            if (std::filesystem::last_write_time(dir_entry) > agentWriteTime) {
                agentFileName = dir_entry.path().string();
                agentWriteTime = std::filesystem::last_write_time(dir_entry);
            }
        }
        std::cout << "Loading model " << agentFileName << "..." << std::endl;
        std::string global_step_string = PPOUtils::getLoadFromSteps(agentFileName, "PPO_Agent_");
        // Update global step to previous 
        m_global_step = (PPOUtils::isNumber(global_step_string)) ? stoi(global_step_string) : 0;
        std::cout << "Continuing training from step " << m_global_step << std::endl;
        torch::load(m_agent, agentFileName);
    }

    // Load optimizer from file
    if (std::distance(std::filesystem::directory_iterator(optimSavePath), std::filesystem::directory_iterator{}) == 0) {
        std::cout << "No previous optimizer checkpoint found at " << modelSavePath << ", initializing new optimizer!" << std::endl;
    }
    else {
        std::string optimFileName;
        std::filesystem::file_time_type optimWriteTime;
        // directory_iterator can be iterated using a range-for loop
        for (auto const& dir_entry : std::filesystem::directory_iterator{ optimSavePath })
        {
            if (std::filesystem::last_write_time(dir_entry) > optimWriteTime) {
                optimFileName = dir_entry.path().string();
                optimWriteTime = std::filesystem::last_write_time(dir_entry);
            }
        }
        std::cout << "Loading optimizer " << optimFileName << "..." << std::endl;
        torch::load(*m_optimizer, optimFileName);
    }

}



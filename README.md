# PPO-LibTorch
PPO-LibTorch is a fully open-source and robust implementation of Proximal Policy Optimization converted from the wonderful [ICLR Blog Post by Huang, et al.](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## Features
PPO-LibTorch offers a few unique features:

1. ***A Custom Threadpool is Used to Support PPO's Vectorized Architecture***.
    * Avoids the slowdowns associated with spinning up and down new threads.
    * Allows the use of a single learner that collects samples and learns from multiple environments.
    * Execution of these environments all originates from a single process requiring no IPC.
      
2. ***Fully Customizable and Explained Hyperparameters***.
    * Hyperparameters can be customized via a config file, no need to recompile for each change.
    * Hyperparameters are clearly explained on this page's wiki.
      
3. ***Easy Integration of Custom Environments***
    * Environments provided can be copied and modified for your needs.
    * No need for you to modify the existing PPO code.
      
4. ***Extremely Readable Implementation***
    * This implementation follows the Python implementation as closely as possible.
    * Almost everything is commented so you can tell what each portion is doing.

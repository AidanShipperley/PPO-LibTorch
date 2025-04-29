#ifndef CARTPOLE_H
#define CARTPOLE_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <tuple>
#include <torch/torch.h>

class CartPole {

  private:
      float gravity;
      float mass_cart;
      float mass_pole;
      float total_mass;
      float length;
      float polemass_length;
      float force_mag;
      float tau;

      // Angle at which to fail the episode
      float theta_threshold_radians;
      float x_threshold;

      // Random number generation
      std::mt19937 gen;
      std::uniform_real_distribution<float> dis;

  protected:

      float randomUniform();

  public:

      // State vector [x, x_dot, theta, theta_dot]
      std::vector<float> state;

      // If true, the episode has terminated
      bool terminated;

	  // Logging Parameters
	  int64_t episode_length;
	  float episode_reward;

	  // Construction and Destruction
	  CartPole(int64_t seed);
	  ~CartPole();

      /*
      The C++ Standard Library's <random> header provides types like std::mt19937 
      and std::random_device that model stateful random number generators. They 
      maintain internal state that changes each time a random number is generated. 
      Because copying that state could lead to repeated random numbers, these types 
      are non-copyable, which means that any class that includes them as members also 
      becomes non-copyable, unless a custom copy constructor is provided.

      The simplest solution is to explicitly delete the copy constructor 
      (and the copy assignment operator as well). This will give you a clear 
      error if you try to copy a CartPole object.

      If you need CartPole objects to be movable but not copyable 
      (for example, to store them in a std::vector), is to leave 
      the copy constructor deleted and provide a move constructor instead
      */
      CartPole(const CartPole&) = delete;
      CartPole& operator=(const CartPole&) = delete;
      CartPole(CartPole&&) = default;
      CartPole& operator=(CartPole&&) = default;

	  // Required Env Functions
	  std::tuple<std::vector<float>, float, bool, bool> step(const int64_t& action);
	  std::vector<float> reset();

};


#endif
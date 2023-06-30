#ifndef MOUNTAIN_CAR_H
#define MOUNTAIN_CAR_H

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <tuple>
#include <torch/torch.h>

class MountainCar {
	public:

		// Environment Parameters
		float min_position;
		float max_position;
		float max_speed;
		float goal_position;
		float goal_velocity;

		float force;
		float gravity;

		std::vector<float> low;
		std::vector<float> high;

		std::vector<float> state;

		// Algo Parameters
		std::vector<int> actionSpace;

		// Logging Parameters
		int episode_length;
		float episode_reward;


		// Construction and Destruction
		MountainCar();
		~MountainCar();

		// Required Env Functions
		std::tuple<std::vector<float>, float, bool, bool> step(int action);
		std::vector<float> reset();

		// Other Env Functions
		torch::Tensor getActionMask();

	protected:

		std::vector<float> randomUniform(float low, float high);



};


#endif
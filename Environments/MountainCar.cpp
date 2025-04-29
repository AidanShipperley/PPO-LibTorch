
#include "MountainCar.h"

MountainCar::MountainCar() {

	min_position = -1.2f;
	max_position = 0.6f;
	max_speed = 0.07f;
	goal_position = 0.5f;
	goal_velocity = 0.0f;

	force = 0.001f;
	gravity = 0.0025f;

	low = { min_position, -max_speed };
	high = { max_position, max_speed };

	episode_length = 0;
	episode_reward = 0.0f;

	state = {};

}

MountainCar::~MountainCar() {
}


std::tuple<std::vector<float>, float, bool, bool> MountainCar::step(const int64_t& action) {

	float& position = state[0];
	float& velocity = state[1];

	velocity += (action - 1.0f) * force + std::cos(3.0f * position) * (-gravity);
	velocity = std::clamp(velocity, -max_speed, max_speed);

	position += velocity;
	position = std::clamp(position, min_position, max_position);

	if (position == min_position && velocity < 0.0f) {
		velocity = 0.0f;
	}

	bool terminated = position >= goal_position && velocity >= goal_velocity;
	
	float reward = -1.0f;

	// Add reward for achieving higher velocities
	//reward += std::abs(velocity) / max_speed;

	// These are for logging purposes
	episode_length += 1;
	episode_reward += reward;

	return { std::vector<float>{position, velocity}, reward, terminated, false };

}


std::vector<float> MountainCar::reset() {

	state = randomUniform(-0.6f, -0.4f);
	episode_length = 0;
	episode_reward = 0.0f;

	return state;
}

torch::Tensor MountainCar::getActionMask() {

	// Don't really need an action mask for this env
	// 
	// Create empty mask
	torch::Tensor mask = torch::ones(3, torch::kBool);

	return mask;
}

std::vector<float> MountainCar::randomUniform(float low, float high) {

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(low, high);

	std::vector<float> result = { dis(gen), 0.0f };
	return result;

}




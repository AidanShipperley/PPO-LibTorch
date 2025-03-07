#include "CartPole.h"

CartPole::CartPole(int64_t seed)
    : gen(seed), dis(-0.05f, 0.05f) { // initialize gen with rd() and dis with desired range

    gravity = 9.8f;
    mass_cart = 1.0f;
    mass_pole = 0.1f;
    total_mass = mass_pole + mass_cart;
    length = 0.5f; // actually half the pole's length
    polemass_length = mass_pole * length;
    force_mag = 10.0f;
    tau = 0.02f; // seconds between state updates

    // Angle at which to fail the episode
    theta_threshold_radians = 12 * 2 * M_PI / 360;
    x_threshold = 2.4f;

    // State vector [x, x_dot, theta, theta_dot]
    state = { 0, 0, 0, 0 };

    // If true, the episode has terminated
    terminated = false;

	episode_length = 0;
	episode_reward = 0.0f;

}

CartPole::~CartPole() {

}

std::vector<float> CartPole::reset() {

    for (size_t i = 0; i < 4; i++) {
        state[i] = randomUniform();
    }
    terminated = false;

    episode_length = 0;
    episode_reward = 0.0f;

    return state;
}

std::tuple<std::vector<float>, float, bool, bool> CartPole::step(const int64_t& action) {

    float x = state[0];
    float x_dot = state[1];
    float theta = state[2];
    float theta_dot = state[3];

    float force = force_mag;
    if (action == 0) {
        force = -force;
    }

    float cos_theta = std::cos(theta);
    float sin_theta = std::sin(theta);

    float temp = (force + polemass_length * theta_dot * theta_dot * sin_theta) / total_mass;
    float theta_acc = (gravity * sin_theta - cos_theta * temp) / (length * (4.0f / 3.0f - mass_pole * cos_theta * cos_theta / total_mass));
    float x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass;

    x = x + tau * x_dot;
    x_dot = x_dot + tau * x_acc;
    theta = theta + tau * theta_dot;
    theta_dot = theta_dot + tau * theta_acc;

    state[0] = x;
    state[1] = x_dot;
    state[2] = theta;
    state[3] = theta_dot;

    if (x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians) {
        terminated = true;
    }

    float reward;

    if (!terminated) {
        reward = 1.0f;
    }
    else {
        //std::cout << "episode terminated with length " << episode_length << std::endl;
        reward = -1.0f;
    }

    episode_length += 1;
    episode_reward += reward;

    return { state, reward, terminated, false };
}


float CartPole::randomUniform() {

    return dis(gen);

}
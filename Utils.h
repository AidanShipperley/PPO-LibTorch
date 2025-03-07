#ifndef PPO_UTILS_H
#define PPO_UTILS_H

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include "third_party/tomlplusplus/toml.hpp"


class PPOUtils {
public:

    static float getVectorMean(std::vector<float> vector);
    static std::string formatString(std::string& str);
    static std::string getLoadFromSteps(std::string const& str, std::string const& fileHead);
    static bool isNumber(const std::string& s);

};


// Fixed size circular buffer for env stat tracking
// Fixed memory usage: The buffer never grows beyond the initial allocation
// O(1) operations: All operations (add, calculate averages) are constant time
// No shifting of elements: When removing old data, we don't need to move memory around
// Pre-computed statistics: Averages are maintained incrementally, avoiding full recalculation
// No allocations during training: The buffer is allocated once at initialization
class CircularBuffer {
private:
    std::vector<float> m_reward_data;
    std::vector<int64_t> m_length_data;
    size_t m_capacity;
    size_t m_size;
    size_t m_head;  // Position to write next element
    double m_reward_sum;
    double m_length_sum;

public:
    CircularBuffer(size_t capacity)
        : m_reward_data(capacity),
          m_length_data(capacity),
          m_capacity(capacity),
          m_size(0),
          m_head(0),
          m_reward_sum(0.0),
          m_length_sum(0.0) {}

    void add(float reward, int64_t length) {
        // If we're at capacity, subtract the value that will be overwritten
        if (m_size == m_capacity) {
            m_reward_sum -= m_reward_data[m_head];
            m_length_sum -= m_length_data[m_head];
        } else {
            m_size++;
        }

        // Add new values
        m_reward_data[m_head] = reward;
        m_length_data[m_head] = length;
        m_reward_sum += reward;
        m_length_sum += length;

        // Move head pointer (wrapping around if needed)
        m_head = (m_head + 1) % m_capacity;
    }

    bool empty() const { return m_size == 0; }
    size_t size() const { return m_size; }
    
    float avgReward() const { 
        return m_size > 0 ? m_reward_sum / m_size : 0.0f; 
    }
    
    double avgLength() const { 
        return m_size > 0 ? m_length_sum / m_size : 0.0; 
    }
};



#endif // PPO_UTILS_H
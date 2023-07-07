#ifndef PPO_UTILS_H
#define PPO_UTILS_H

#include <vector>
#include <string>
#include <algorithm>


class PPOUtils {
public:

    static float getVectorMean(std::vector<float> vector);
    static std::string formatString(std::string& str);
    static std::string getLoadFromSteps(std::string const& str, std::string const& fileHead);
    static bool isNumber(const std::string& s);

};





#endif // PPO_UTILS_H
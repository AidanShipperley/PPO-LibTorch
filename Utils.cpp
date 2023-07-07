// Utils.cpp
#include "Utils.h"


float PPOUtils::getVectorMean(std::vector<float> vector) {

    float runningTotal = 0.f;
    for (int i = 0; i < vector.size(); i++) {
        runningTotal += vector[i];
    }

    return runningTotal / vector.size();

}


std::string PPOUtils::formatString(std::string& str) {
    // Remove whitespace
    str.erase(std::remove(str.begin(), str.end(), ' '), str.end());

    // Make all lowercase
    std::transform(str.begin(), str.end(), str.begin(),
        [](unsigned char c) { return std::tolower(c); } // correct
    );

    return str;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// getLoadFromSteps() -> Returns std::string
// ---------------------------------
// Attempts to grab the checkpoint steps from the 
// filename to continue training. Only affects logging
// and when to terminate training, as training ends
// when this is >= m_num_timesteps.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
std::string PPOUtils::getLoadFromSteps(std::string const& str, std::string const& fileHead) {
    char const* digits = "0123456789";
    std::size_t const n = str.find(fileHead);
    std::size_t const newN = n + fileHead.length();
    if (newN != std::string::npos)
    {
        std::size_t const m = str.find_first_not_of(digits, newN);
        return str.substr(newN, m != std::string::npos ? m - newN : m);
    }
    return std::string();
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// isNumber() -> Returns bool
// ---------------------------------
// Given a string, this will determine if the string
// is only comprised of digits. Use when casting 
// strings to ints as a safeguard.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bool PPOUtils::isNumber(const std::string& s) {
    return !s.empty() && std::find_if(s.begin(),
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}
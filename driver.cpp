// driver.cpp : Defines the entry point for the application.
//

//#include "PPO_MultiDiscrete.h"
#include "PPO_Discrete.h"

int main()
{

	try {
		PPO_Discrete algo;

		algo.train();
	}
	catch (const std::exception& ex) {
		std::cerr << "Error Occured: " << ex.what() << std::endl;
	}

	return 0;

}
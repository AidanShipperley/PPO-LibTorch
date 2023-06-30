// driver.cpp : Defines the entry point for the application.
//

#include <iostream>
//#include "PPO_MultiDiscrete.h"
#include "PPO_Discrete.h"

int main()
{

	PPO_Discrete algo;

	algo.train();

	return 0;

}
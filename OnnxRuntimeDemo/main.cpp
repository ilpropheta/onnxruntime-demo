#include <iostream>
#include "Linear.h"
#include "ResNet.h"
#include "MobileNet.h"

using namespace std;

int main()
{
	try
	{
		std::cout << "ResNet 50:\n";
		Demo::RunResNet();
		
		std::cout << "\n\n";
		
		std::cout << "MobileNet ssd lite:\n";
		Demo::RunMobileNet();
	}
	catch (const exception& e)
	{
		cout << e.what() << "\n";
	}
}
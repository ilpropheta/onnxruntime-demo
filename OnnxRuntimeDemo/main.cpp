#include <iostream>
#include "Linear.h"
#include "ResNet.h"
#include "MobileNet.h"

using namespace std;

int main()
{
	try
	{
		Demo::RunLinearRegression();
		//Demo::RunResNet();
		//Demo::RunMobileNet();
	}
	catch (const exception& e)
	{
		cout << e.what() << "\n";
	}
}
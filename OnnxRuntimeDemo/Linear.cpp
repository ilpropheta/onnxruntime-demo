#include "Linear.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <iostream>

using namespace std;

void Demo::RunLinearRegression()
{
	// gives access to the underlying API (you can optionally customize log)
	// you can create one environment per process (each environment manages an internal thread pool)
	Ort::Env env;

	// creates an inference session for a certain model
	Ort::Session session{ env, LR"(data\linear.onnx)", Ort::SessionOptions{nullptr} };

	// Ort::Session gives access to input and output information:
	// - count
	// - name
	// - shape and type
	std::cout << "Number of model inputs: " << session.GetInputCount() << "\n";
	std::cout << "Number of model outputs: " << session.GetOutputCount() << "\n";

	// you can customize how allocation works. Let's just use a default allocator provided by the library
	Ort::AllocatorWithDefaultOptions allocator;
	// get input and output names
	auto* inputName = session.GetInputName(0, allocator);
	std::cout << "Input name: " << inputName << "\n";

	auto* outputName = session.GetOutputName(0, allocator);
	std::cout << "Output name: " << outputName << "\n";

	// get input shape
	auto inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	// set some input values
	std::vector<float> inputValues = { 4, 5, 6 };

	// where to allocate the tensors
	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	// create the input tensor (this is not a deep copy!)
	auto inputOnnxTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(), inputShape.data(), inputShape.size());

	// the API needs the array of inputs you set and the array of outputs you get
	array inputNames = { inputName };
	array outputNames = { outputName };

	// finally run the inference!
	auto outputValues = session.Run(
		Ort::RunOptions{ nullptr }, // e.g. set a verbosity level only for this run
		inputNames.data(), &inputOnnxTensor, 1, // input to set
		outputNames.data(), 1);					// output to take 

	// extract first (and only) output
	auto& output1 = outputValues[0];
	const auto* floats = output1.GetTensorMutableData<float>();
	const auto floatsCount = output1.GetTensorTypeAndShapeInfo().GetElementCount();

	// just print the output values
	std::copy_n(floats, floatsCount, ostream_iterator<float>(cout, " "));


	// closing boilerplate
	allocator.Free(inputName);
	allocator.Free(outputName);
}

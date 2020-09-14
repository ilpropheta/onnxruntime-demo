#include "ResNet.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xview.hpp>
#include "Utils.h"

using namespace std;

static const int ImageWidth = 224;
static const int ImageHeight = 224;

auto PreprocessImageForResNet(const cv::Mat& frame)
{
	auto mat = Utils::ResizeToFloat(frame, { ImageWidth, ImageHeight });
	xt::xarray<float> tens = xt::adapt(reinterpret_cast<float*>(mat.data), { 1, ImageWidth, ImageHeight, 3 });
	tens = xt::transpose(std::move(tens), { 0, 3, 1, 2 });

	xt::xarray<float> mean_vec{ 0.485f, 0.456f, 0.406f };
	xt::xarray<float> std_vec{ 0.229f, 0.224f, 0.225f };
	xt::xarray<float> norm_data = xt::zeros<float>({ 1, 3, ImageWidth, ImageHeight });

	for (ptrdiff_t i = 0; i < 3; ++i)
	{
		//    norm_data[:,i,:,:]                            = (img_data[:,i,:,:]/255 - mean_vec[i])/std_vec[i]
		view(norm_data, xt::all(), i, xt::all(), xt::all()) = (view(tens, xt::all(), i, xt::all(), xt::all()) - mean_vec[i]) / std_vec[i];
	}

	return norm_data;
}

void Demo::RunResNet()
{
	Ort::Env env;
	Ort::Session session{ env, LR"(data\resnet50v2.onnx)", Ort::SessionOptions{nullptr} };
	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	// classes for inference
	const auto classes = Utils::ReadClasses(R"(data\ImagenetClasses.txt)");

	std::vector inputNames = Utils::OnnxGetInputNames(session);
	std::vector outputNames = Utils::OnnxGetOutputNames(session);

	const auto inputsAsConstCharPtr = Utils::MakeConstCharPtrVector(inputNames);
	const auto outputsAsConstCharPtr = Utils::MakeConstCharPtrVector(outputNames);
	
	// iterate over the .jpg contained in the input folder
	Utils::ForEachImage(".jpg", "data", [&](cv::Mat& image, const auto& imagePath) {

		// PreprocessImageForMobileNet data and return an xtensor-specific tensor
		auto inputTensor = PreprocessImageForResNet(image);
		std::vector<int64_t> inputShape(inputTensor.shape().begin(), inputTensor.shape().end());
		auto onnxInputTensor = Ort::Value::CreateTensor<float>(memoryInfo, 
			inputTensor.data(), inputTensor.size(), 
			inputShape.data(), inputShape.size());

		auto onnxOutputTensor = session.Run(Ort::RunOptions{ nullptr },
			inputsAsConstCharPtr.data(), &onnxInputTensor, inputsAsConstCharPtr.size(), 
			outputsAsConstCharPtr.data(), outputsAsConstCharPtr.size());

		auto outputTensor = Utils::AsSpan(onnxOutputTensor[0]);

		Utils::softmax(outputTensor);

		const auto idx = distance(begin(outputTensor), max_element(begin(outputTensor), end(outputTensor)));
		cout << imagePath << " class: " << classes[idx] << " with % " << outputTensor[idx] * 100 << "\n";
	});
}
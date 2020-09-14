#include "MobileNet.h"
#include <iostream>
#include <filesystem>
#include <onnxruntime_cxx_api.h>
#include "Box.h"
#include "span.h"
#include "Utils.h"
#include "DrawingUtils.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>

using namespace std;
using namespace Utils;

// MobileNet preprocess and postprocess code has been adapted from the awesome work of our friends at Unimore:
// https://github.com/ceccocats/tkDNN/blob/cnet/src/MobilenetDetection.cpp
// (part of the tkDNN library)

static const size_t N_COORDS = 4;
static const size_t N_SSDSPEC = 6;
static const float centerVariance = 0.1f;
static const float sizeVariance = 0.2f;

struct SSDSpec
{
	int featureSize = 0;
	int shrinkage = 0;
	int boxWidth = 0;
	int boxHeight = 0;
	int ratio1 = 0;
	int ratio2 = 0;

	void setAll(int feature_size, int shrinkage, int box_width, int box_height, int ratio1, int ratio2)
	{
		this->featureSize = feature_size;
		this->shrinkage = shrinkage;
		this->boxWidth = box_width;
		this->boxHeight = box_height;
		this->ratio1 = ratio1;
		this->ratio2 = ratio2;
	}
};

std::vector<float> GenerateSDDPriors(const std::array<SSDSpec, N_SSDSPEC>& specs, float imageSize)
{
	const auto nPriors = std::accumulate(begin(specs), end(specs), 0, [](auto curr, const auto& spec) {
		return curr + spec.featureSize * spec.featureSize * 6;
	});
	
	std::vector<float> priors(N_COORDS * nPriors);

	int i_prio = 0;
	float scale, x_center, y_center, h, w, ratio;
	int min, max;
	for (int i = 0; i < N_SSDSPEC; i++)
	{
		scale = imageSize / (float)specs[i].shrinkage;
		min = specs[i].boxHeight > specs[i].boxWidth ? specs[i].boxWidth : specs[i].boxHeight;
		max = specs[i].boxHeight < specs[i].boxWidth ? specs[i].boxWidth : specs[i].boxHeight;
		for (int j = 0; j < specs[i].featureSize; j++) 
		{
			for (int k = 0; k < specs[i].featureSize; k++) 
			{
				//small sized square box
				x_center = (k + 0.5f) / scale;
				y_center = (j + 0.5f) / scale;
				h = w = min / imageSize;

				priors[i_prio * N_COORDS + 0] = x_center;
				priors[i_prio * N_COORDS + 1] = y_center;
				priors[i_prio * N_COORDS + 2] = w;
				priors[i_prio * N_COORDS + 3] = h;
				++i_prio;

				//big sized square box
				h = w = sqrtf(static_cast<float>(max) * min) / imageSize;

				priors[i_prio * N_COORDS + 0] = x_center;
				priors[i_prio * N_COORDS + 1] = y_center;
				priors[i_prio * N_COORDS + 2] = w;
				priors[i_prio * N_COORDS + 3] = h;
				++i_prio;

				//change h/w ratio of the small sized box
				h = w = min / imageSize;
				ratio = sqrtf(static_cast<float>(specs[i].ratio1));
				priors[i_prio * N_COORDS + 0] = x_center;
				priors[i_prio * N_COORDS + 1] = y_center;
				priors[i_prio * N_COORDS + 2] = w * ratio;
				priors[i_prio * N_COORDS + 3] = h / ratio;
				++i_prio;

				priors[i_prio * N_COORDS + 0] = x_center;
				priors[i_prio * N_COORDS + 1] = y_center;
				priors[i_prio * N_COORDS + 2] = w / ratio;
				priors[i_prio * N_COORDS + 3] = h * ratio;
				++i_prio;

				ratio = sqrtf(static_cast<float>(specs[i].ratio2));
				priors[i_prio * N_COORDS + 0] = x_center;
				priors[i_prio * N_COORDS + 1] = y_center;
				priors[i_prio * N_COORDS + 2] = w * ratio;
				priors[i_prio * N_COORDS + 3] = h / ratio;
				++i_prio;

				priors[i_prio * N_COORDS + 0] = x_center;
				priors[i_prio * N_COORDS + 1] = y_center;
				priors[i_prio * N_COORDS + 2] = w / ratio;
				priors[i_prio * N_COORDS + 3] = h * ratio;
				++i_prio;
			}
		}
	}

	std::transform(begin(priors), end(priors), begin(priors), [](auto prior) {
		return std::clamp(prior, 0.0f, 1.0f);
	});

	return priors;
}

void ConvertLocationsToBoxesAndCenter(size_t nPriors, span<float> locations_h, span<const float> priors)
{
	float cur_x, cur_y;
	for (auto i = 0u; i < nPriors; i++)
	{
		locations_h[i * N_COORDS + 0] = locations_h[i * N_COORDS + 0] * centerVariance * priors[i * N_COORDS + 2] + priors[i * N_COORDS + 0];
		locations_h[i * N_COORDS + 1] = locations_h[i * N_COORDS + 1] * centerVariance * priors[i * N_COORDS + 3] + priors[i * N_COORDS + 1];
		locations_h[i * N_COORDS + 2] = exp(locations_h[i * N_COORDS + 2] * sizeVariance) * priors[i * N_COORDS + 2];
		locations_h[i * N_COORDS + 3] = exp(locations_h[i * N_COORDS + 3] * sizeVariance) * priors[i * N_COORDS + 3];

		cur_x = locations_h[i * N_COORDS + 0];
		cur_y = locations_h[i * N_COORDS + 1];

		locations_h[i * N_COORDS + 0] = cur_x - locations_h[i * N_COORDS + 2] / 2;
		locations_h[i * N_COORDS + 1] = cur_y - locations_h[i * N_COORDS + 3] / 2;
		locations_h[i * N_COORDS + 2] = cur_x + locations_h[i * N_COORDS + 2] / 2;
		locations_h[i * N_COORDS + 3] = cur_y + locations_h[i * N_COORDS + 3] / 2;
	}
}

std::vector<float> priors;

void InitPriors()
{
	static const float imageSize = 512;
	std::array<SSDSpec, N_SSDSPEC> specs{};
	specs[0].setAll(32, 16, 60, 105, 2, 3);
	specs[1].setAll(16, 32, 105, 150, 2, 3);
	specs[2].setAll(8, 64, 150, 195, 2, 3);
	specs[3].setAll(4, 100, 195, 240, 2, 3);
	specs[4].setAll(2, 150, 240, 285, 2, 3);
	specs[5].setAll(1, 300, 285, 330, 2, 3);
	priors = GenerateSDDPriors(specs, imageSize);
}

std::vector<Box> MobileNetPostprocess(span<float> confidences_h, span<float> locations_h, int classes, cv::Size originalSize, float confThreshold)
{
	const auto nPriors = priors.size() / 4u;

	ConvertLocationsToBoxesAndCenter(nPriors, locations_h, priors);

	int width = originalSize.width;
	int height = originalSize.height;
	const float IoUThreshold = 0.45f;

	std::vector<Box> detected;

	float* conf_per_class;
	for (int i = 1; i < classes; i++) 
	{
		conf_per_class = &confidences_h[i * nPriors];
		std::vector<Box> boxes;
		for (int j = 0; j < nPriors; j++) 
		{

			if (conf_per_class[j] > confThreshold) 
			{
				Box b;
				b.cl = i;
				b.prob = conf_per_class[j];
				b.x = locations_h[j * N_COORDS + 0];
				b.y = locations_h[j * N_COORDS + 1];
				b.w = locations_h[j * N_COORDS + 2];
				b.h = locations_h[j * N_COORDS + 3];

				boxes.push_back(b);
			}
		}
		std::sort(boxes.begin(), boxes.end(), GreaterProbability);

		std::vector<Box> remaining;
		while (boxes.size() > 0) 
		{
			remaining.clear();

			Box b;
			b.cl = boxes[0].cl - 1;             //remove background class
			b.prob = boxes[0].prob;
			b.x = boxes[0].x * width;
			b.y = boxes[0].y * height;
			b.w = boxes[0].w * width - b.x;     //convert from x1 to width
			b.h = boxes[0].h * height - b.y;    //convert from y1 to height
			detected.push_back(b);
			for (size_t j = 1; j < boxes.size(); j++) 
			{
				if (IoU(boxes[0], boxes[j]) <= IoUThreshold) {
					remaining.push_back(boxes[j]);
				}
			}
			boxes = remaining;
		}
	}

	return detected;
}

tuple<int, int, float> SoftMax(int candidates, int classNum, span<float> scores)
{
	int maxidx = 0;
	auto maxval = 0.0f;
	auto bestcandidate = 0;

	for (auto i = 0; i < candidates; ++i)
	{
		auto disp = classNum * i;
		// remember to skip the first column because it's the background!
		//                                      v
		span subv(scores.data() + disp + 0, scores.data() + disp + classNum);
		softmax(subv);
		auto subvmax = max_element(begin(subv), end(subv));

		if (*subvmax > maxval)
		{
			maxidx = static_cast<int>(distance(begin(subv), subvmax));
			maxval = *subvmax;
			bestcandidate = i;
		}
	}

	return { maxidx, bestcandidate, maxval };
}

std::vector<Box> Postprocess(Ort::Value& scoresTensor, Ort::Value& boxesTensor, cv::Size originalSize, float confThreshold)
{
	auto scores = Utils::AsSpan(scoresTensor);
	auto boxes = Utils::AsSpan(boxesTensor);

	const auto scoresShape = scoresTensor.GetTensorTypeAndShapeInfo().GetShape();
	const auto candidates = static_cast<int>(scoresShape[1]);
	const auto classes = static_cast<int>(scoresShape[2]);

	auto [maxidx, bestcandidate, maxval] = SoftMax(candidates, classes, scores);

	auto scoresXTensor = xt::adapt(scores.data(), scoresShape);
	scoresXTensor = xt::transpose(scoresXTensor, { 0, 2, 1 });

	return MobileNetPostprocess(scores, boxes, classes, originalSize, confThreshold);
}

static auto PreprocessImageForMobileNet(const cv::Mat& frame)
{
	auto prepocessed = RemoveMeanDivideByStd(frame, { 512, 512 });
	xt::xarray<float> tens = xt::adapt(reinterpret_cast<float*>(prepocessed.data), { 1, prepocessed.cols, prepocessed.rows, 3 });
	return xt::eval(xt::transpose(std::move(tens), { 0, 3, 1, 2 }));
}

void Demo::RunMobileNet()
{
	InitPriors();
	
	Ort::Env env;
	Ort::Session session{ env, LR"(data\mobileNet.onnx)", Ort::SessionOptions{nullptr} };
	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	auto outShape = GetOutputShape(session, 0);
	const auto classes = static_cast<int>(outShape[2]);

	const auto colors = Drawing::MakeColors(classes);
	
	std::vector inputNames = Utils::OnnxGetInputNames(session);
	std::vector outputNames = Utils::OnnxGetOutputNames(session);

	const auto inputsAsConstCharPtr = Utils::MakeConstCharPtrVector(inputNames);
	const auto outputsAsConstCharPtr = Utils::MakeConstCharPtrVector(outputNames);

	// iterate over the .jpg contained in the input folder
	ForEachImage(".jpg", "data", [&](cv::Mat& frame, const auto& imagePath) {

		// since the model itself is generated from PyTorch, we need to convert
		// the frame from OpenCV format {w, h, channels} to PyTorch tensor format {batch, channels, w, h} that is {1, 3, w, h }
		auto inputTensor = PreprocessImageForMobileNet(frame);

		try
		{
			std::vector<int64_t> inputShape(inputTensor.shape().begin(), inputTensor.shape().end());
			auto onnxInputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
				inputTensor.data(), inputTensor.size(),
				inputShape.data(), inputShape.size());

			auto onnxOutputTensor = session.Run(Ort::RunOptions{ nullptr },
				inputsAsConstCharPtr.data(), &onnxInputTensor, inputsAsConstCharPtr.size(),
				outputsAsConstCharPtr.data(), outputsAsConstCharPtr.size());

			auto detectedBoundingBoxes = Postprocess(onnxOutputTensor[0], onnxOutputTensor[1], frame.size(), 0.3f);

			// save output images with detected bounding boxes
			Drawing::DrawBoundingBoxes(frame, detectedBoundingBoxes, colors);
			const auto outputFileName = (std::filesystem::path("outdata") / imagePath.filename()).string();
			if (cv::imwrite(outputFileName, frame))
				std::cout << "output saved into " << outputFileName << "\n\n";
		}
		catch (const exception& ex)
		{
			std::cout << ex.what() << "\n";
		}
	});
}
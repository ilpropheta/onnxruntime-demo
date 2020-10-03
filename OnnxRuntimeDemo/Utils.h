#pragma once
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include "span.h"

namespace Ort
{
	struct Session;
	struct Value;
}

namespace Utils
{
	cv::Mat ResizeToFloat(const cv::Mat& frame, const cv::Size& size, float alpha = 1.0f / 255.0f, float beta = 0.0f, cv::InterpolationFlags interpolation = cv::InterpolationFlags::INTER_LINEAR);
	cv::Mat RemoveMeanDivideByStd(const cv::Mat& frame, cv::Size size);
	
	std::vector<std::string> ReadClasses(const char* fileName);
	const std::vector<std::string>& GetCoco2017Classes();
	
	template<typename Action>
	void ForEachImage(const char* extension, const char* imgPath, Action action)
	{
		for (auto& p : std::filesystem::directory_iterator(imgPath))
		{
			if (p.is_regular_file() && p.path().extension() == extension)
			{
				auto image = cv::imread(p.path().string(), cv::IMREAD_COLOR);
				action(image, p.path());
			}
		}
	}

	template<typename Action>
	void ForEachImage_N(const char* extension, const char* imgPath, int N, Action action)
	{
		for (auto& p : std::filesystem::directory_iterator(imgPath))
		{
			if (N-- == 0)
				break;
			if (p.is_regular_file() && p.path().extension() == extension)
			{
				auto image = cv::imread(p.path().string(), cv::IMREAD_COLOR);
				action(image, p.path());
			}
		}
	}

	template<typename T>
	void softmax(span<T> data)
	{
		const auto m = *std::max_element(std::begin(data), std::end(data));
		const auto sum = std::accumulate(std::begin(data), std::end(data), T{}, [=](auto s, auto i) {
			return s + std::exp(i - m);
		});
		const auto offset = m + std::log(sum);
		std::transform(std::begin(data), std::end(data), std::begin(data), [=](auto i) {
			return std::exp(i - offset);
		});
	}

	template<typename T>
	T overlap(T x1, T w1, T x2, T w2)
	{
		T l1 = x1 - w1 / 2;
		T l2 = x2 - w2 / 2;
		T left = l1 > l2 ? l1 : l2;
		T r1 = x1 + w1 / 2;
		T r2 = x2 + w2 / 2;
		T right = r1 < r2 ? r1 : r2;
		return right - left;
	}

	std::string OnnxGetInputName(Ort::Session& session, size_t index);
	std::string OnnxGetOutputName(Ort::Session& session, size_t index);
	std::vector<std::string> OnnxGetInputNames(Ort::Session& session);
	std::vector<std::string> OnnxGetOutputNames(Ort::Session& session);
	std::vector<const char*> MakeConstCharPtrVector(span<std::string> strings);
	std::vector<std::int64_t> GetOutputShape(Ort::Session& session, size_t index);
	
	span<float> AsSpan(Ort::Value& tensor);
}

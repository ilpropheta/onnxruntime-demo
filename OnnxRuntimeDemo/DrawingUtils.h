#pragma once
#include <opencv2\core\types.hpp>
#include "Box.h"

namespace Drawing
{
	std::array<cv::Scalar, 256> MakeColors(int classes);
	cv::Mat DrawBoundingBoxes(cv::Mat frame, const std::vector<Utils::Box>& detected, const std::array<cv::Scalar, 256>& colors);
}
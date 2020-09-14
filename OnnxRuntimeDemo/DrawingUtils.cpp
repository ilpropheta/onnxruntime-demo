#include "DrawingUtils.h"
#include <opencv2/core/mat.hpp>
#include "Utils.h"

static float getColor(int c, int x, int max)
{
	float _colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
	float ratio = (static_cast<float>(x) / max) * 5;
	const auto i = static_cast<int>(floor(ratio));
	const auto j = static_cast<int>(ceil(ratio));
	ratio -= i;
	return (1 - ratio) * _colors[i % 6][c % 3] + ratio * _colors[j % 6][c % 3];
}

std::array<cv::Scalar, 256> Drawing::MakeColors(int classes)
{
	std::array<cv::Scalar, 256> colors{};
	for (int c = 0; c < classes; c++)
	{
		const int offset = c * 123457 % classes;
		const float r = getColor(2, offset, classes);
		const float g = getColor(1, offset, classes);
		const float b = getColor(0, offset, classes);
		colors[c] = cv::Scalar(static_cast<int>(255.0 * b), static_cast<int>(255.0 * g), static_cast<int>(255.0 * r));
	}
	return colors;
}

static std::tuple<int, int, int, int> CalculateMobileNetBounds(const Utils::Box& b)
{
	const auto x0 = static_cast<int>(b.x);
	const auto x1 = static_cast<int>(b.x + b.w);
	const auto y0 = static_cast<int>(b.y);
	const auto y1 = static_cast<int>(b.y + b.h);
	return { x0, x1, y0, y1 };
};

cv::Mat Drawing::DrawBoundingBoxes(cv::Mat frame, const std::vector<Utils::Box>& detected, const std::array<cv::Scalar, 256>& colors)
{
	static const float font_scale = 0.5;
	static const int thickness = 2;
	static const auto& classes = Utils::GetCoco2017Classes();
	
	for (const auto& b : detected)
	{
		const auto [x0, x1, y0, y1] = CalculateMobileNetBounds(b);
		const auto& classToDraw = classes[b.cl];

		// draw rectangle
		rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), colors[b.cl], 2);

		// draw label
		int baseline = 0;
		const cv::Size text_size = getTextSize(classToDraw, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
		rectangle(frame, cv::Point(x0, y0), cv::Point((x0 + text_size.width - 2), (y0 - text_size.height - 2)), colors[b.cl], -1);
		putText(frame, classToDraw, cv::Point(x0, (y0 - (baseline / 2))), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
	}
	return frame;
}
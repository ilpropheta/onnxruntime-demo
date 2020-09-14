#include "Box.h"

#include "Utils.h"

Utils::GroundTruthBox Utils::ClearTruth(GroundTruthBox g)
{
	g.unique_truth_index = -1;
	g.truth_flag = 0;
	g.max_IoU = 0;
	return g;
}

float Utils::IoU(const Utils::Box& a, const Utils::Box& b)
{
	const float max_x = a.x > b.x ? a.x : b.x;
	const float max_y = a.y > b.y ? a.y : b.y;
	const float min_w = a.w < b.w ? a.w : b.w;
	const float min_h = a.h < b.h ? a.h : b.h;

	const float ao_w = min_w - max_x > 0 ? min_w - max_x : 0;
	const float ao_h = min_h - max_y > 0 ? min_h - max_y : 0;

	const float area_overlap = ao_w * ao_h;
	const float area_0_w = a.w - a.x > 0 ? a.w - a.x : 0;
	const float area_0_h = a.h - a.y > 0 ? a.h - a.y : 0;

	const float area_1_w = b.w - b.x > 0 ? b.w - b.x : 0;
	const float area_1_h = b.h - b.y > 0 ? b.h - b.y : 0;

	const float area_0 = area_0_h * area_0_w;
	const float area_1 = area_1_h * area_1_w;

	const float iou = area_overlap / (area_0 + area_1 - area_overlap + 1e-5f);
	return iou;
}

float Utils::Intersection(const Utils::GroundTruthBox& a, const Utils::GroundTruthBox& b)
{
	const float w = overlap(a.x, a.w, b.x, b.w);
	const float h = overlap(a.y, a.h, b.y, b.h);
	if (w < 0 || h < 0)
		return 0;
	const float area = w * h;
	return area;
}

float Utils::Union(const Utils::GroundTruthBox& a, const Utils::GroundTruthBox& b)
{
	const float i = Intersection(a, b);
	const float u = a.w * a.h + b.w * b.h - i;
	return u;
}

float Utils::IoU(const Utils::GroundTruthBox& a, const Utils::GroundTruthBox& b)
{
	const float I = Intersection(a, b);
	const float U = Union(a, b);
	if (I == 0 || U == 0)
		return 0;
	return I / U;
}

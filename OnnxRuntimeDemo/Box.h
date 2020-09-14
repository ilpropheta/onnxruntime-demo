#pragma once
#include <string>
#include <vector>
#include <tuple>

namespace Utils
{
    struct Box
    {
        int cl = 0;
        float prob = 0;
        float x = 0;
        float y = 0;
        float w = 0;
        float h = 0;
    };

    inline bool GreaterProbability(const Box& a, const Box& b)
    {
        return (a.prob > b.prob);
    }

    inline auto AsTuple(const Box& b)
    {
        return std::tie(b.cl, b.prob, b.x, b.y, b.w, b.h);
    }

    inline bool operator==(const Box& b1, const Box& b2)
    {
        return AsTuple(b1) == AsTuple(b2);
    }

    struct GroundTruthBox : Box
    {
        int unique_truth_index = -1;
        int truth_flag = 0;
        float max_IoU = 0;
    };

    GroundTruthBox ClearTruth(GroundTruthBox g);

    inline auto AsTuple(const GroundTruthBox& b)
    {
        return std::tuple_cat(
            AsTuple(static_cast<const Box&>(b)),
            std::tie(b.unique_truth_index, b.truth_flag, b.max_IoU));
    }

    inline bool operator==(const GroundTruthBox& b1, const GroundTruthBox& b2)
    {
        return AsTuple(b1) == AsTuple(b2);
    }

    struct FrameDetectionInfo
    {
        std::string LabelFilename;
        std::string ImageFilename;
        std::vector<GroundTruthBox> GroundTruthBoxes;
        std::vector<GroundTruthBox> DetectedBoxes;
    };

    float IoU(const Box& a, const Box& b);

    float Intersection(const GroundTruthBox& a, const GroundTruthBox& b);

    float Union(const GroundTruthBox& a, const GroundTruthBox& b);

    float IoU(const GroundTruthBox& a, const GroundTruthBox& b);
}
#include "Utils.h"
#include <fstream>
#include <onnxruntime_cxx_api.h>

cv::Mat Utils::ResizeToFloat(const cv::Mat& frame, const cv::Size& size, float alpha, float beta, cv::InterpolationFlags interpolation)
{
	cv::Mat dst;
	cv::resize(frame, dst, size, 0.0f, 0.0f, interpolation);
	dst.convertTo(dst, CV_32FC3, alpha, beta);
	return dst;
}

cv::Mat Utils::RemoveMeanDivideByStd(const cv::Mat& frame, cv::Size size)
{
	cv::Mat dst;
	resize(frame, dst, size);
	dst.convertTo(dst, CV_32FC3, 1.0f, -127.0f);
	dst.convertTo(dst, CV_32FC3, 1.0f / 128.0f, 0.0f);
	return dst;
}

std::vector<std::string> Utils::ReadClasses(const char* fileName)
{
	std::ifstream file(fileName);
	std::vector<std::string> out;
	std::string str;
	while (std::getline(file, str))
	{
		out.push_back(move(str));
	}
	return out;
}

const std::vector<std::string>& Utils::GetCoco2017Classes()
{
	static const std::vector<std::string> names = {
		"person" , "bicycle" , "car" , "motorbike" , "aeroplane" , "bus" ,
		"train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "stop sign" ,
		"parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" ,
		"elephant" , "bear" , "zebra" , "giraffe" , "backpack" , "umbrella" , "handbag" ,
		"tie" , "suitcase" , "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" ,
		"baseball bat" , "baseball glove" , "skateboard" , "surfboard" , "tennis racket" ,
		"bottle" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , "banana" ,
		"apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" , "pizza" ,
		"donut" , "cake" , "chair" , "sofa" , "pottedplant" , "bed" , "diningtable" ,
		"toilet" , "tvmonitor" , "laptop" , "mouse" , "remote" , "keyboard" ,
		"cell phone" , "microwave" , "oven" , "toaster" , "sink" , "refrigerator" ,
		"book" , "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" };

	return names;
}

struct OrtDeleter
{
	OrtAllocator& allocator;
	void operator()(void* ptr) const
	{
		allocator.Free(&allocator, ptr);
	}
};

using OrtStringOwner = std::unique_ptr<char, OrtDeleter>;

template<typename MemFun>
static std::string OnnxGetString(MemFun memf, Ort::Session& session, size_t index)
{
	Ort::AllocatorWithDefaultOptions allocator;
	const auto chars = std::invoke(memf, session, index, allocator);
	const OrtStringOwner owner{ chars, OrtDeleter{ *allocator } };
	return owner.get();
}

std::string Utils::OnnxGetInputName(Ort::Session& session, size_t index)
{
	return OnnxGetString(&Ort::Session::GetInputName, session, index);
}

std::string Utils::OnnxGetOutputName(Ort::Session& session, size_t index)
{
	return OnnxGetString(&Ort::Session::GetOutputName, session, index);
}

template<typename Getter, typename CountGetter>
std::vector<std::string> OnnxGetNames(Getter getter, CountGetter countGetter, Ort::Session& session)
{
	std::vector<std::string> out;
	const auto nInputs = std::invoke(countGetter, session);
	for (auto i = 0; i < nInputs; ++i)
	{
		out.push_back(std::invoke(getter, session, i));
	}
	return out;
}

std::vector<std::string> Utils::OnnxGetInputNames(Ort::Session& session)
{
	return OnnxGetNames(OnnxGetInputName, &Ort::Session::GetInputCount, session);
}

std::vector<std::string> Utils::OnnxGetOutputNames(Ort::Session& session)
{
	return OnnxGetNames(OnnxGetOutputName, &Ort::Session::GetOutputCount, session);
}

std::vector<const char*> Utils::MakeConstCharPtrVector(span<std::string> strings)
{
	std::vector<const char*> out(strings.size());
	std::transform(begin(strings), end(strings), begin(out), [](const auto& str)
	{
		return str.c_str();
	});
	return out;
}

std::vector<std::int64_t> Utils::GetOutputShape(Ort::Session& session, size_t index)
{
	return session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

Utils::span<float> Utils::AsSpan(Ort::Value& tensor)
{
	return {tensor.GetTensorMutableData<float>(), tensor.GetTensorTypeAndShapeInfo().GetElementCount() };
}
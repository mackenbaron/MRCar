#pragma once
#include "mropencv.h"
#include "fstream"
#include "sstream"
using namespace std;
static const char* kDefaultAnnPath = "ann.xml";
class AnnInstance
{
public:
	static AnnInstance* instance_;
	static AnnInstance* instance()
	{
		if (!instance_)
		{
			instance_ = new AnnInstance();
		}
		return instance_;
	}
	cv::Ptr<cv::ml::ANN_MLP> ann_;
	AnnInstance()
	{
		ann_ = ml::ANN_MLP::load<ml::ANN_MLP>(kDefaultAnnPath);
	}
	int predict(Mat &feature)
	{
		return ann_->predict(feature);
	}
};
AnnInstance*AnnInstance::instance_ = nullptr;
int predictbyann(Mat &img)
{
	cv::Mat feature = features(img, 10);
	return static_cast<int>(AnnInstance::instance()->predict(feature));
}
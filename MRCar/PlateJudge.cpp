#include "PlateJudge.h"
#include "mrcharutil.h"
static const char* kDefaultSvmPath = "svm.xml";
CPlateJudge::CPlateJudge()
{
	svm_ = ml::SVM::load<ml::SVM>(kDefaultSvmPath);
}


CPlateJudge::~CPlateJudge()
{
}

bool CPlateJudge::plateJudge(vector<Mat> &inputs, vector<Mat> &results)
{
	for (auto it = inputs.begin(); it != inputs.end(); it++)
	{
		int res = -1;
		plateJudge((*it), res);
		if (res == 1)
			results.push_back((*it));
	}
	return true;
}



int CPlateJudge::plateJudge(Mat &input, int &result)
{
	Mat featues;
//	imwrite("img.jpg", input);
	getHistogramFeatures(input, featues);
	float response = svm_->predict(featues);
	result = (int)response;
	return 0;
}
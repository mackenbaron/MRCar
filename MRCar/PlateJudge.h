#pragma once
#include "mropencv.h"
#include "vector"
using namespace std;
class CPlateJudge
{
public:
	CPlateJudge();
	~CPlateJudge();
	bool plateJudge(vector<Mat> &inputs,vector<Mat> &results);
	int plateJudge(Mat &input, int &result);
	Ptr<ml::SVM> svm_;
};


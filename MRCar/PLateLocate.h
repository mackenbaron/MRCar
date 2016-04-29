#pragma once
#include "mropencv.h"
#include "vector"
using namespace std;
class CPLateLocate
{
public:
	CPLateLocate();
	~CPLateLocate();
public:
	int plateLocate(Mat &img, vector<Mat> &result);
	bool verifySize(RotatedRect &rr);
	float m_angle=60;
};


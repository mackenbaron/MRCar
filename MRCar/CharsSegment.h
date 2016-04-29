#pragma once
#include "mropencv.h"
#include "vector"
using namespace std;
class CCharsSegment
{
public:
	CCharsSegment();
	~CCharsSegment();
	int charSegment(Mat &input, vector<Mat> &results);
	bool verifySizes(Mat r);
	int getSpecificRect(vector<Rect> rects);
	Rect getChineseRect(const Rect &r);
	void RebuildRect(const vector<Rect> rects,vector<Rect>&outRect,int specIndex);
	Mat preprocessChar(Mat &in);
};


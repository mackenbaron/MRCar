#include "PlateRecognition.h"


CPlateRecognition::CPlateRecognition()
{
}


CPlateRecognition::~CPlateRecognition()
{
}

string CPlateRecognition::plateRecognition(Mat &in)
{
	vector<Mat> resutlVec;
	CPLateLocate pl;
	pl.plateLocate(in, resutlVec);
	//Mat img = imread("img.jpg");
	//resutlVec.push_back(img);
	CPlateJudge judge;
	vector<Mat> judges;
	judge.plateJudge(resutlVec, judges);
	for (auto it = judges.begin(); it != judges.end(); it++)
	{
		//		imwrite("chars_segment.jpg", *it);
		//		imshow("judge", *it);
		vector<Mat> chars;
		CCharsSegment cs;
		Mat topplate = in(Rect(0, 0, it->cols, it->rows));
		addWeighted(topplate, 0, *it, 1, 0, topplate);
		cs.charSegment(*it, chars);
		string license;
		CharsIdentify ci;
		for (auto it = chars.begin(); it != chars.end(); it++)
		{
			license += ci.identify(*it);
		}
		return license;
	}
}
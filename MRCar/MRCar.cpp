#include "mropencv.h"
#include "PlateRecognition.h"
#include "tick_meter.hpp"
#include "CvxText.h"

int main()
{
 	TickMeter tm;
 	tm.start();
	Mat img = imread("test.jpg");
	CPlateRecognition pr;
	string lic = pr.plateRecognition(img);
	tm.stop();
	cout <<"ºÄÊ±:"<< tm.getTimeMilli() <<"ms"<< endl;
	CvxText text("simhei.ttf");
	resize(img, img, Size(960, 720));
	float p = 0.5;
	text.setFont(NULL, NULL, NULL, &p);
	text.putText(&(IplImage)img, lic.c_str(), cvPoint(10, 50), CV_RGB(255, 0, 0));
	imshow("img", img);
	waitKey();
	return 0;
}
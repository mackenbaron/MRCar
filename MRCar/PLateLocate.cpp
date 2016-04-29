#include "PLateLocate.h"

#define SOBEL_SCALE 1
CPLateLocate::CPLateLocate()
{
}


CPLateLocate::~CPLateLocate()
{
}

bool CPLateLocate::verifySize(RotatedRect &rr)
{
	float error = 0.9;
	float aspect = 4;
	int min = 44 * 14 * 1;
	int max = 44 * 14 * 30;
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;
	int area = rr.size.height*rr.size.width;
	float r = (float)rr.size.width / rr.size.height;
	if (r < 1)
	{
		r = (float)rr.size.height / rr.size.width;
	}
	if ((area<min || area>max) || (r<rmin) || r>rmax)
		return false;
	else
		return true;
}
int CPLateLocate::plateLocate(Mat &img, vector<Mat> &result)
{
	Mat src_blur, src_gray;
	Mat grad;
	GaussianBlur(img, src_blur, Size(5, 5), 0, 0, 4);
	cvtColor(src_blur, src_gray, CV_RGB2GRAY);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(src_gray, grad_x, img.depth(), 1, 0, 3);
	Sobel(src_gray, grad_y, img.depth(), 0, 1, 3);
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 1, abs_grad_y, 1, 0, grad);
	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
	Mat img_threshold;
	threshold(grad, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	morphologyEx(img_threshold, img_threshold, MORPH_CLOSE, element);
	vector<vector<Point>>contours;
	findContours(img_threshold, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//drawContours(img, contours, -1, Scalar(0, 0, 255));
	
	int t = 0;
	vector<RotatedRect> rects;
	for (auto it = contours.begin(); it != contours.end(); )
	{
		RotatedRect rr = minAreaRect(Mat(*it));
		if (!verifySize(rr))
		{
			it = contours.erase(it);
		}
		else
		{
			it++;
			rects.push_back(rr);
		}
	}
	int k = 1;
	for (int i = 0; i < rects.size(); i++)
	{
		RotatedRect minRect = rects[i];
		float r = (float)minRect.size.width / (float)minRect.size.height;
		float angle = minRect.angle;
		Size rect_size = minRect.size;
		if (r < 1)
		{
			angle = angle + 90;
			std::swap(rect_size.width, rect_size.height);
		}
		if (angle - m_angle<0 && angle + m_angle>0)
		{
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
			Mat img_rotated;
			warpAffine(img, img_rotated, rotmat, img.size(), CV_INTER_CUBIC);
			Mat resultMat;
			Mat img_crop;
			getRectSubPix(img, rect_size, minRect.center, img_crop);
			resize(img_crop, resultMat, Size(136,36), 0, 0, INTER_CUBIC);
			result.push_back(resultMat);
		}
	}
// 	for (auto it = result.begin(); it != result.end(); it++)
// 	{
// 		imshow("result", *it);
// 		imwrite("img.jpg", *it);
// 		waitKey(1);
// 	}
	return 0;
}

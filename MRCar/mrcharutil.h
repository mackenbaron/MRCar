#ifndef MRCHAR_UTIL_H
#define MRCHAR_UTIL_H
#include "mropencv.h"
#include "vector"
using namespace std;

Mat features(Mat in, int sizeData);
Mat ProjectedHistogram(Mat img, int t);
void getHistogramFeatures(const cv::Mat& image, cv::Mat& features);

#endif
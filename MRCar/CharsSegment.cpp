#include "CharsSegment.h"


CCharsSegment::CCharsSegment()
{
}


CCharsSegment::~CCharsSegment()
{
}
int ThresholdOtsu(Mat mat) {
	int height = mat.rows;
	int width = mat.cols;

	// histogram
	float histogram[256] = { 0 };
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			unsigned char p = (unsigned char)((mat.data[i * mat.step[0] + j]));
			histogram[p]++;
		}
	}
	// normalize histogram
	int size = height * width;
	for (int i = 0; i < 256; i++) {
		histogram[i] = histogram[i] / size;
	}

	// average pixel value
	float avgValue = 0;
	for (int i = 0; i < 256; i++) {
		avgValue += i * histogram[i];
	}

	int thresholdV;
	float maxVariance = 0;
	float w = 0, u = 0;
	for (int i = 0; i < 256; i++) {
		w += histogram[i];
		u += i * histogram[i];

		float t = avgValue * w - u;
		float variance = t * t / (w * (1 - w));
		if (variance > maxVariance) {
			maxVariance = variance;
			thresholdV = i;
		}
	}

	return thresholdV;
}
bool clearLiuDing(Mat &img)
{
	vector<int> nJump;
	int whiteCount = 0;
	const int x = 7;
	for (int i = 0; i < img.rows; i++)
	{
		int jumpCount = 0;
		for (int j = 0; j < img.cols - 1; j++)
		{
			if (img.at<char>(i, j) != img.at<char>(i, j + 1))
				jumpCount++;
			if (img.at<uchar>(i, j) == 255)				
				whiteCount++;
		}
		nJump.push_back(jumpCount);
	}
	int iCount = 0;
	for (int i = 0; i < img.rows; i++)
	{
		if (nJump[i] >= 16 && nJump[i] <= 45)
			iCount++;
	}
	if (iCount*1.0 / img.rows < 0.4)
		return false;
	if (whiteCount * 1.0 / (img.rows * img.cols) < 0.15 ||
		whiteCount * 1.0 / (img.rows * img.cols) > 0.50) {
		return false;
	}
	for (int i = 0; i < img.rows; i++)
	{
		if (nJump[i] <= x)
		{
			for (int j = 0; j < img.cols; j++)
				img.at<char>(i, j) = 0;
		}
	}
	return true;
}
int CCharsSegment::charSegment(Mat &input, vector<Mat> &results)
{
	Mat gray;
	cvtColor(input, gray, CV_RGB2GRAY);
	Mat img_threshold;
	int threadHoldV = ThresholdOtsu(gray(Rect_<double>(gray.cols*0.1,gray.rows*0.1,gray.cols*0.8,gray.rows*0.8)));
	threshold(gray, img_threshold, threadHoldV, 255, CV_THRESH_BINARY);
	clearLiuDing(img_threshold);
	Mat img_contours;
	img_threshold.copyTo(img_contours);
	vector<vector<Point>>contours;
	vector<Rect> vecRect;
	findContours(img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (auto it = contours.begin(); it != contours.end();it++)
	{
		Rect mr = boundingRect(Mat(*it));
		Mat auxRoi(img_threshold, mr);
		if (verifySizes(auxRoi))
			vecRect.push_back(mr);		
	}
	if (vecRect.size() == 0)
		return 3;
	vector<Rect>sortedRect(vecRect);
	sort(sortedRect.begin(), sortedRect.end(),[](const Rect&r1, const Rect &r2){return r1.x < r2.x; });
	int specIndex = getSpecificRect(sortedRect);
	Rect chinsesRect;
	if (specIndex < sortedRect.size())
		chinsesRect = getChineseRect(sortedRect[specIndex]);
	else
		return 4;
	vector<Rect>newSortedRect;
	newSortedRect.push_back(chinsesRect);
	RebuildRect(sortedRect, newSortedRect, specIndex);
	if (newSortedRect.size() == 0)
		return 5;
	for (auto i=0; i<newSortedRect.size(); i++)
	{
		Rect mr = newSortedRect[i];
		Mat auxRoi(gray, mr);
		Mat newRoi;
		threshold(auxRoi, newRoi, 5, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		newRoi=preprocessChar(newRoi);
		results.push_back(newRoi);
	}
	return 0;
}


bool CCharsSegment::verifySizes(Mat r)
{
	float aspect = 45.0f / 90.0f;
	float charAspect = (float)r.cols / (float)r.rows;
	float error = 0.7f;
	float minHeight = 10.f;
	float maxHeight = 35.f;
	float minAspect = 0.05f;
	float maxAspect = aspect + aspect*error;
	int area = countNonZero(r);
	int bbArea = r.cols*r.rows;
	int perPixels = area / bbArea;
	if (perPixels <= 1 && charAspect > minAspect&&charAspect, maxAspect&&r.rows >= minHeight&&r.rows < maxHeight)
		return true;
	else
		return false;
}

int CCharsSegment::getSpecificRect(vector<Rect> rects)
{
	vector<int>positions;
	int maxHeight = 0;
	int maxWidth = 0;
	for (auto i = 0; i < rects.size(); i++)
	{
		positions.push_back(rects[i].x);
		if (rects[i].height>maxHeight)
			maxHeight = rects[i].height;
		if (rects[i].width > maxWidth)
			maxWidth = rects[i].width;
		int specIndex = 0;
		for (auto i = 0; i < rects.size(); i++)
		{
			Rect mr = rects[i];
			int midx = mr.x+ mr.width / 2;
			if ((mr.width>maxWidth*0.8 || mr.height > maxHeight*0.8) && (midx<int(136 / 7) * 2 && midx>int(136 / 7) * 1))
				specIndex = i;
		}
		return specIndex;
	}
}

Rect CCharsSegment::getChineseRect(const Rect &r)
{
	int height = r.height;
	float newwidth = r.width*1.15f;
	int x = r.x;
	int y = r.y;
	int newx = x - int(newwidth*1.15);
	newx = newx > 0 ? newx: 0;
	return Rect(newx, y, int(newwidth), height);
}

void CCharsSegment::RebuildRect(const vector<Rect> rects, vector<Rect>&outRect, int specIndex)
{
	int count = 6;
	for (auto i = specIndex; i < rects.size(); i++, count--)
	{
		outRect.push_back(rects[i]);
	}
}

Mat CCharsSegment::preprocessChar(Mat &in)
{
	int h = in.rows;
	int w = in.cols;
	int charSize = 20;
	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
	transformMat.at<float>(1, 2) = float(m / 2 - h / 2);
	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat,warpImage.size(), INTER_LINEAR,
		BORDER_CONSTANT, Scalar(0));
	Mat out;
	resize(warpImage, out, Size(charSize, charSize));
	return out;
}
#include "windows.h"
#include "mropencv.h"
#include "vector"
#include "fstream"
#include "mrutil.h"
using namespace std;

static const char* kDefaultSvmPath = "svm.xml";
static const char* kDefaultAnnPath = "ann.xml";
Ptr<ml::SVM> svm_ = ml::SVM::load<ml::SVM>(kDefaultSvmPath);
cv::Ptr<cv::ml::ANN_MLP> ann_ = ml::ANN_MLP::load<ml::ANN_MLP>(kDefaultAnnPath);
map<string, string> provincemapping_;
static const char *kChars[] = {
	"0", "1", "2",
	"3", "4", "5",
	"6", "7", "8",
	"9",
	/*  10  */
	"A", "B", "C",
	"D", "E", "F",
	"G", "H", /* {"I", "I"} */
	"J", "K", "L",
	"M", "N", /* {"O", "O"} */
	"P", "Q", "R",
	"S", "T", "U",
	"V", "W", "X",
	"Y", "Z",
	/*  24  */
	"zh_cuan", "zh_e", "zh_gan",
	"zh_gan1", "zh_gui", "zh_gui1",
	"zh_hei", "zh_hu", "zh_ji",
	"zh_jin", "zh_jing", "zh_jl",
	"zh_liao", "zh_lu", "zh_meng",
	"zh_min", "zh_ning", "zh_qing",
	"zh_qiong", "zh_shan", "zh_su",
	"zh_sx", "zh_wan", "zh_xiang",
	"zh_xin", "zh_yu", "zh_yu1",
	"zh_yue", "zh_yun", "zh_zang",
	"zh_zhe"
	/*  31  */
};

std::string utf8_to_gbk(const char* utf8) {
	int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
	wchar_t* wszGBK = new wchar_t[len + 1];
	memset(wszGBK, 0, len * 2 + 2);
	MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wszGBK, len);
	len = WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, NULL, 0, NULL, NULL);
	char* szGBK = new char[len + 1];
	memset(szGBK, 0, len + 1);
	WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, szGBK, len, NULL, NULL);
	std::string strTemp(szGBK);
	if (wszGBK)
		delete[] wszGBK;
	if (szGBK)
		delete[] szGBK;
	return strTemp;
}

void init()
{
	ifstream pf("province_mapping");
	if (!pf)
	{
		pf.close();
		return;
	}
	string line;
	while (!pf.eof())
	{
		getline(pf, line);
		if (line.empty())
			continue;
		vector<string> ret;
		split(line, string(" "), &ret);
		string chname = line.substr(line.size() - 3, line.size());
		provincemapping_[ret[0]] = utf8_to_gbk(chname.c_str());
	}
}

bool verifySize(RotatedRect &rr)
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

void plateDetect(Mat &img, vector<Mat> &result)
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
	for (auto it = contours.begin(); it != contours.end();)
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
		if (angle - 60<0 && angle + 60>0)
		{
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);
			Mat img_rotated;
			warpAffine(img, img_rotated, rotmat, img.size(), CV_INTER_CUBIC);
			Mat resultMat;
			Mat img_crop;
			getRectSubPix(img, rect_size, minRect.center, img_crop);
			resize(img_crop, resultMat, Size(136, 36), 0, 0, INTER_CUBIC);
			result.push_back(resultMat);
		}
	}
}

float countOfBigValue(Mat &mat, int iValue) {
	float iCount = 0.0;
	if (mat.rows > 1) {
		for (int i = 0; i < mat.rows; ++i) {
			if (mat.data[i * mat.step[0]] > iValue) {
				iCount += 1.0;
			}
		}
		return iCount;

	}
	else {
		for (int i = 0; i < mat.cols; ++i) {
			if (mat.data[i] > iValue) {
				iCount += 1.0;
			}
		}

		return iCount;
	}
}

Mat ProjectedHistogram(Mat img, int t) {
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j < sz; j++) {
		Mat data = (t) ? img.row(j) : img.col(j);

		//统计这一行或一列中，非零元素的个数，并保存到mhist中

		mhist.at<float>(j) = countOfBigValue(data, 20);
	}

	// Normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);

	//用mhist直方图中的最大值，归一化直方图

	if (max > 0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);

	return mhist;
}

Mat getTheFeatures(Mat in) {
	const int VERTICAL = 0;
	const int HORIZONTAL = 1;

	// Histogram features
	Mat vhist = ProjectedHistogram(in, VERTICAL);
	Mat hhist = ProjectedHistogram(in, HORIZONTAL);

	// Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols;

	Mat out = Mat::zeros(1, numCols, CV_32F);

	// Asign values to feature,样本特征为水平、垂直直方图

	int j = 0;
	for (int i = 0; i < vhist.cols; i++) {
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++) {
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}

	return out;
}

void getHistogramFeatures(const Mat& image, Mat& features) {
	Mat grayImage;
	cvtColor(image, grayImage, CV_RGB2GRAY);

	//grayImage = histeq(grayImage);

	Mat img_threshold;
	threshold(grayImage, img_threshold, 0, 255,
		CV_THRESH_OTSU + CV_THRESH_BINARY);
	features = getTheFeatures(img_threshold);
}

int plateJudgeOneMat(Mat &input, int &result)
{
	Mat featues;
	getHistogramFeatures(input, featues);
	float response = svm_->predict(featues);
	result=(int)response;
	return true;
}

void plateJudge(vector<Mat> &inputs, vector<Mat> &results)
{
	for (auto it = inputs.begin(); it != inputs.end(); it++)
	{
		int res = -1;
		plateJudgeOneMat((*it), res);
		if (res == 1)
			results.push_back((*it));
	}
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

bool verifySizes(Mat r)
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

int getSpecificRect(vector<Rect> rects)
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
			int midx = mr.x + mr.width / 2;
			if ((mr.width>maxWidth*0.8 || mr.height > maxHeight*0.8) && (midx<int(136 / 7) * 2 && midx>int(136 / 7) * 1))
				specIndex = i;
		}
		return specIndex;
	}
}

Rect getChineseRect(const Rect &r)
{
	int height = r.height;
	float newwidth = r.width*1.15f;
	int x = r.x;
	int y = r.y;
	int newx = x - int(newwidth*1.15);
	newx = newx > 0 ? newx : 0;
	return Rect(newx, y, int(newwidth), height);
}

void RebuildRect(const vector<Rect> rects, vector<Rect>&outRect, int specIndex)
{
	int count = 6;
	for (auto i = specIndex; i < rects.size(); i++, count--)
	{
		outRect.push_back(rects[i]);
	}
}

Mat preprocessChar(Mat &in)
{
	int h = in.rows;
	int w = in.cols;
	int charSize = 20;
	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
	transformMat.at<float>(1, 2) = float(m / 2 - h / 2);
	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR,
		BORDER_CONSTANT, Scalar(0));
	Mat out;
	resize(warpImage, out, Size(charSize, charSize));
	return out;
}

int charSegment(Mat &input, vector<Mat> &results)
{
	Mat gray;
	cvtColor(input, gray, CV_RGB2GRAY);
	Mat img_threshold;
	int threadHoldV = ThresholdOtsu(gray(Rect_<double>(gray.cols*0.1, gray.rows*0.1, gray.cols*0.8, gray.rows*0.8)));
	threshold(gray, img_threshold, threadHoldV, 255, CV_THRESH_BINARY);
	clearLiuDing(img_threshold);
	Mat img_contours;
	img_threshold.copyTo(img_contours);
	vector<vector<Point>>contours;
	vector<Rect> vecRect;
	findContours(img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (auto it = contours.begin(); it != contours.end(); it++)
	{
		Rect mr = boundingRect(Mat(*it));
		Mat auxRoi(img_threshold, mr);
		if (verifySizes(auxRoi))
			vecRect.push_back(mr);
	}
	if (vecRect.size() == 0)
		return 3;
	vector<Rect>sortedRect(vecRect);
	sort(sortedRect.begin(), sortedRect.end(), [](const Rect&r1, const Rect &r2){return r1.x < r2.x; });
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
	for (auto i = 0; i < newSortedRect.size(); i++)
	{
		Rect mr = newSortedRect[i];
		Mat auxRoi(gray, mr);
		Mat newRoi;
		threshold(auxRoi, newRoi, 5, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		newRoi = preprocessChar(newRoi);
		results.push_back(newRoi);
	}
	return 0;
}

#define HORIZONTAL 1
#define VERTICAL 0

Mat CutTheRect(Mat &in, Rect &rect) {
	int size = in.cols;  // (rect.width>rect.height)?rect.width:rect.height;
	Mat dstMat(size, size, CV_8UC1);
	dstMat.setTo(Scalar(0, 0, 0));

	int x = (int)floor((float)(size - rect.width) / 2.0f);
	int y = (int)floor((float)(size - rect.height) / 2.0f);

	//把rect中的数据 考取到dstMat的中间

	for (int i = 0; i < rect.height; ++i) {

		//宽

		for (int j = 0; j < rect.width; ++j) {
			dstMat.data[dstMat.step[0] * (i + y) + j + x] =
				in.data[in.step[0] * (i + rect.y) + j + rect.x];
		}
	}

	//
	return dstMat;
}

Rect GetCenterRect(Mat &in) {
	Rect _rect;

	int top = 0;
	int bottom = in.rows - 1;

	//上下

	for (int i = 0; i < in.rows; ++i) {
		bool bFind = false;
		for (int j = 0; j < in.cols; ++j) {
			if (in.data[i * in.step[0] + j] > 20) {
				top = i;
				bFind = true;
				break;
			}
		}
		if (bFind) {
			break;
		}

		//统计这一行或一列中，非零元素的个数

	}
	for (int i = in.rows - 1;
		i >= 0;
		--i) {
		bool bFind = false;
		for (int j = 0; j < in.cols; ++j) {
			if (in.data[i * in.step[0] + j] > 20) {
				bottom = i;
				bFind = true;
				break;
			}
		}
		if (bFind) {
			break;
		}

		//统计这一行或一列中，非零元素的个数

	}

	//左右

	int left = 0;
	int right = in.cols - 1;
	for (int j = 0; j < in.cols; ++j) {
		bool bFind = false;
		for (int i = 0; i < in.rows; ++i) {
			if (in.data[i * in.step[0] + j] > 20) {
				left = j;
				bFind = true;
				break;
			}
		}
		if (bFind) {
			break;
		}

		//统计这一行或一列中，非零元素的个数

	}
	for (int j = in.cols - 1;
		j >= 0;
		--j) {
		bool bFind = false;
		for (int i = 0; i < in.rows; ++i) {
			if (in.data[i * in.step[0] + j] > 20) {
				right = j;
				bFind = true;

				break;
			}
		}
		if (bFind) {
			break;
		}

		//统计这一行或一列中，非零元素的个数

	}

	_rect.x = left;
	_rect.y = top;
	_rect.width = right - left + 1;
	_rect.height = bottom - top + 1;

	return _rect;
}

Mat features(Mat in, int sizeData) {

	//抠取中间区域

	Rect _rect = GetCenterRect(in);

	Mat tmpIn = CutTheRect(in, _rect);
	// Mat tmpIn = in.clone();
	// Low data feature
	Mat lowData;
	resize(tmpIn, lowData, Size(sizeData, sizeData));

	// Histogram features
	Mat vhist = ProjectedHistogram(lowData, VERTICAL);
	Mat hhist = ProjectedHistogram(lowData, HORIZONTAL);

	// Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;

	Mat out = Mat::zeros(1, numCols, CV_32F);
	// Asign values to

	// feature,ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量

	int j = 0;
	for (int i = 0; i < vhist.cols; i++) {
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++) {
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x < lowData.cols; x++) {
		for (int y = 0; y < lowData.rows; y++) {
			out.at<float>(j) += (float)lowData.at < unsigned
				char >(x, y);
			j++;
		}
	}
	return out;
}

string identify(Mat &in)
{
	Mat feature = features(in, 10);

	auto index = static_cast<int>(ann_->predict(feature));
	if (index < 34)
	{
		return kChars[index];
	}
	else
	{
		return provincemapping_[kChars[index]];
	}
}

vector<string> RecognizeOneImage(Mat &img)
{
	init();
	vector<Mat> resutlVec;
	plateDetect(img, resutlVec);
	vector<Mat> judges;
	plateJudge(resutlVec, judges);
	vector<string> licenses;
	for (auto it = judges.begin(); it != judges.end(); it++)
	{
		vector<Mat> chars;
		Mat topplate = img(Rect(0, 0, it->cols, it->rows));
		addWeighted(topplate, 0, *it, 1, 0, topplate);
		charSegment(*it, chars);
		string license;
		for (auto it = chars.begin(); it != chars.end(); it++)
		{
			license += identify(*it);
		}
		licenses.push_back(license);
	}
	return licenses;
}

bool getfiles(string strDir, vector<string> &files)
{
	WIN32_FIND_DATA FindData;
	HANDLE hError;
	string file2find = strDir + "*.*";
	hError = FindFirstFile((LPCTSTR)file2find.c_str(), &FindData);
	if (hError == INVALID_HANDLE_VALUE)
	{
		return 0;
	}
	else
	{
		do
		{
			//过滤.和..;“.”代表本级目录“..”代表父级目录;
			if (lstrcmp(FindData.cFileName, TEXT(".")) == 0 || lstrcmp(FindData.cFileName, TEXT("..")) == 0)
			{
				continue;
			}
			if (!(FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				files.push_back(FindData.cFileName);
				
			}
			if (FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			{
				//files.push_back(WChar2Ansi(FindData.cFileName));
				continue;
			}
		} while (::FindNextFile(hError, &FindData));
	}
	FindClose(hError);
	return 0;
}

float Testnight(string directory)
{
	float accuracy = 0;
	vector<string>files;
	getfiles(directory, files);
	for (auto file : files)
	{
		Mat img = imread(directory + "/" + file);
		resize(img, img, Size(480, 640));
		vector<string> plates=RecognizeOneImage(img);
		for (auto plate : plates)
			cout << plate << endl;
		imshow("img", img);
		waitKey(1);
	}
	return accuracy;
}
int main1()
{
	cout<<Testnight("E:\\PatternRecognition\\PlateRecognition\\dataset\\night\\")<<endl;
	return 0;
}

int main()
{
	Mat img = imread("test.jpg");
	vector<string>licenses = RecognizeOneImage(img);
	for (auto license : licenses)
		cout<<license<<endl;
	imshow("img", img);
	waitKey();
	return 0;
}

int main2()
{
	Mat img = imread("test.jpg");
	init();
	vector<Mat> resutlVec;
	plateDetect(img, resutlVec);
	vector<Mat> judges;
	plateJudge(resutlVec, judges);
	for (auto it = judges.begin(); it != judges.end(); it++)
	{
		vector<Mat> chars;
		Mat topplate = img(Rect(0, 0, it->cols, it->rows));
		addWeighted(topplate, 0, *it, 1, 0, topplate);
		charSegment(*it, chars);
		string license;
		for (auto it = chars.begin(); it != chars.end(); it++)
		{
			license += identify(*it);
		}
		cout << license << endl;
		imshow("plates", *it);
		waitKey();
	}	return 0;
}
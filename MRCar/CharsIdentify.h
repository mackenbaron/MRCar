#pragma once
#include "mropencv.h"
#include "vector"
#include "map"
using namespace std;


class CharsIdentify
{
public:
	CharsIdentify();
	~CharsIdentify();
	string identify(Mat &in);
	map<string, string> provincemapping_;
};

#pragma once
#include "PLateLocate.h"
#include "PlateJudge.h"
#include "CharsSegment.h"
#include "CharsIdentify.h"

class CPlateRecognition
{
public:
	CPlateRecognition();
	~CPlateRecognition();
	string plateRecognition(Mat &in);
};


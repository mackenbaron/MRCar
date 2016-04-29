#include <windows.h>
#include "CharsIdentify.h"
#include "mrcharutil.h"
#include "fstream"
#include "mrutil.h"
#define ANN_CNN_SWITCH 0
#if ANN_CNN_SWITCH
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "charscnn.h"
#else
#include "charsann.h"
#endif

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

CharsIdentify::CharsIdentify()
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
		string chname = line.substr(line.size() - 3,line.size());
		provincemapping_[ret[0]] = utf8_to_gbk(chname.c_str());
	}
}


CharsIdentify::~CharsIdentify()
{
}

string CharsIdentify::identify(Mat &in)
{
#if ANN_CNN_SWITCH
	auto index = predictbycnn(in);
#else
	auto index = static_cast<int>(predictbyann(in));
#endif
	if (index < 34)
	{
		return kChars[index];
	}
	else
	{
		return provincemapping_[kChars[index]];
	}
}
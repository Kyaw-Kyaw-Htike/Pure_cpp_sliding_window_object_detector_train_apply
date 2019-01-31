// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "NMS_naive.h"

void NMS_naive::suppress(const std::vector<cv::Rect>& dr, const std::vector<float>& ds, std::vector<cv::Rect>& dr_nms, std::vector<float>& ds_nms)
{
	dr_nms = dr;
	ds_nms = ds;
}

// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "opencv2/opencv.hpp"

class NMS_Base
{
public:
	virtual ~NMS_Base();
	virtual void suppress(const std::vector<cv::Rect> &dr,
		const std::vector<float> &ds, std::vector<cv::Rect> &dr_nms,
		std::vector<float> &ds_nms) = 0;
};
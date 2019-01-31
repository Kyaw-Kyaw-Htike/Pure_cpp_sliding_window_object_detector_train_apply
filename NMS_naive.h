// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "NMS_Base.h"

class NMS_naive : public NMS_Base
{
public:
	void suppress(const std::vector<cv::Rect> &dr,
		const std::vector<float> &ds, std::vector<cv::Rect> &dr_nms,
		std::vector<float> &ds_nms) override;
};
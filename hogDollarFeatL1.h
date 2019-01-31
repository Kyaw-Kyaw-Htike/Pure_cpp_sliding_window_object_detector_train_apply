// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "featL1_Base.h"
#include "hog_dollar_wrap.h"

class hogDollarFeatL1 : public featL1_Base
{
public:
	hogDollarFeatL1() = delete;
	hogDollarFeatL1(bool dalalHog, int shrinkage_ = 8);
	cv::Mat extract(const cv::Mat &img) override;
private:
	hog_dollar_wrap hogObj;
};
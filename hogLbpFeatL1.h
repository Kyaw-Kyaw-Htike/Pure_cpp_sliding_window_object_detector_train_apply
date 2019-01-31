// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once
#include "featL1_Base.h"

#include "hog_dollar_wrap.h"
#include "vl_feat_wrappers.h"

class hogLbpFeatL1 : public featL1_Base
{
public:
	hogLbpFeatL1();
	cv::Mat extract(const cv::Mat &img) override;
private:
	hog_dollar_wrap hogObj;
	vl_lbp_w lbpObj;
};
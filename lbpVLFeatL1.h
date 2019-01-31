// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "featL1_Base.h"
#include "vl_feat_wrappers.h"

class lbpVLFeatL1 : public featL1_Base
{
public:
	lbpVLFeatL1();
	cv::Mat extract(const cv::Mat &img) override;
private:
	vl_lbp_w lbpObj;
};

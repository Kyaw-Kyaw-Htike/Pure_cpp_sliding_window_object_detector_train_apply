// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once
#include "featL1_Base.h"
#include "vl_feat_wrappers.h"

class hogVLFeatL1 : public featL1_Base
{
public:
	hogVLFeatL1(int nrows_img, int ncols_img, int nchannels_img);

	cv::Mat extract(const cv::Mat &img) override;
private:
	vl_hog_w hogObj;
};

// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "featL2_Base.h"

class featL2_naive : public featL2_Base
{
public:
	featL2_naive(int ndims_feat_);
	cv::Mat extract(const cv::Mat &patchChannel) override;
};
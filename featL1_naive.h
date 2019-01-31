// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once
#include "featL1_Base.h"


class featL1_naive : public featL1_Base
{
public:
	featL1_naive();
	cv::Mat extract(const cv::Mat &img) override;
};

// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "opencv2/opencv.hpp"

/*
Level 2 feature extraction.
From a patch corresponding to a channel image, it converts to
a feature vector. Simple flattening to make it a vector is just
a special case of this.
*/
class featL2_Base
{
public:
	virtual ~featL2_Base();
	// input patchChannel should be of type CV_32FC(n) where n can vary
	// output of this function should be a row feature vector of type CV_32FC1
	virtual cv::Mat extract(const cv::Mat &patchChannel) = 0;
	int get_ndimsFeat();
protected:
	int ndims_feat;
};
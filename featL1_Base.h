// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "opencv2/opencv.hpp"

/*
Level 1 feature extraction. For mapping input images to
channel like feature "images" (color images are just a special case)
*/
class featL1_Base
{
public:
	virtual ~featL1_Base();
	// input should be of type CV_8UC1 or CV_8UC3; an image
	// output should be of a feature channel image CV_32FC(n)
	// where n can vary
	virtual cv::Mat extract(const cv::Mat &img) = 0;
	int get_shrinkage();
	int get_nchannels();
protected:
	int shrinkage; // must be 1, 4, 8, 12, etc
	int featNChannels;
};
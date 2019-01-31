// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "featL1_naive.h"

featL1_naive::featL1_naive()
{
	shrinkage = 1;
	featNChannels = 1;
}

cv::Mat featL1_naive::extract(const cv::Mat & img)

{
	cv::Mat img_channels;
	cv::cvtColor(img, img_channels, CV_RGB2GRAY);
	img_channels.convertTo(img_channels, CV_32FC1);
	return img_channels;
}

// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "featL2_naive.h"

featL2_naive::featL2_naive(int ndims_feat_)
{
	//ndims_feat = 128 * 64;
	//ndims_feat = 16 * 8 * 31;
	ndims_feat = ndims_feat_;
}

cv::Mat featL2_naive::extract(const cv::Mat & patchChannel)
{
	//cout << "patchChannel num row, col & channels: " << patchChannel.rows << " " << patchChannel.cols << " " << patchChannel.channels() << endl;
	return patchChannel.clone().reshape(1, 1);
}

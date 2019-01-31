// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "NMS_Base.h"
#include "Armadillo"

class NMSGreedy : public NMS_Base
{
	// C++ translation from the NMS matlab code of Tomasz Maliseiwicz who modified
	// Pedro Felzenszwalb's version to speed up. Tested and my C++ version and his matlab 
	// version give exactly
	// the same results. Out of 100 images, C++ version was found to be 1.78992 times 
	// faster than the matlab version on average and 1.74615 times faster on median.
	// This is quite good since the matlab version is already very heavily vectorized
	// and therefore very fast. 

public:
	NMSGreedy();

	void set_thresh(float th);

	void suppress(const std::vector<cv::Rect> &dr,
		const std::vector<float> &ds, std::vector<cv::Rect> &dr_nms,
		std::vector<float> &ds_nms) override;

	void merge_dets(const arma::Mat<float> &dr, const arma::Col<float> &ds,
		arma::Mat<float> &dr_new, arma::Col<float> &ds_new);

private:

	float overlap_thresh;
};
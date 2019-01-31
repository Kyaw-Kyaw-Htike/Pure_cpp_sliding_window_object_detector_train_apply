// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "NMSOpencv.h"

NMSOpencv::NMSOpencv()
{
	use_meanshift = false;
}

NMSOpencv::NMSOpencv(bool use_meanshift_)
{
	use_meanshift = use_meanshift_;
}

void NMSOpencv::suppress(const std::vector<cv::Rect>& dr, const std::vector<float>& ds, std::vector<cv::Rect>& dr_nms, std::vector<float>& ds_nms)
{
	if (!use_meanshift)
	{
		dr_nms = dr;
		cv::groupRectangles(dr_nms, 2);
		ds_nms.resize(dr_nms.size());
		std::fill(ds_nms.begin(), ds_nms.end(), 1);
	}

	else
	{
		dr_nms = dr;
		ds_nms = ds;
		//cv::groupRectangles_meanshift(dr_nms, ds_nms, scales, 0, cv::Size(64, 128));
	}

}

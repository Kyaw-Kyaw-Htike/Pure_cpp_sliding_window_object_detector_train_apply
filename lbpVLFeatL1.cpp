// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "lbpVLFeatL1.h"

lbpVLFeatL1::lbpVLFeatL1() :
	lbpObj(false)
{
	featNChannels = lbpObj.get_num_lbpChannels();
	shrinkage = 8;
}

cv::Mat lbpVLFeatL1::extract(const cv::Mat & img)

{
	cv::Mat img_temp;
	cv::cvtColor(img, img_temp, CV_BGR2GRAY);
	img_temp.convertTo(img_temp, CV_32FC1);

	int nr = img.rows;
	int nc = img.cols;

	std::vector<float> img_vec(nr*nc);
	unsigned int cc = 0;
	float* ptr_row;
	for (size_t i = 0; i < nr; i++)
	{
		ptr_row = img_temp.ptr<float>(i);
		for (size_t j = 0; j < nc; j++)
			img_vec[cc++] = ptr_row[j];
	}

	std::vector<float> H;
	int nr_H, nc_H, nch_H;
	lbpObj.extract_feat(img_vec.data(), nr, nc, H, nr_H, nc_H, nch_H, shrinkage);

	cv::Mat feats_cv(nr_H, nc_H, CV_32FC(nch_H));

	int ss[3] = { nr_H, nc_H, nch_H };
	cv::Mat feats_cv_temp = feats_cv.reshape(1, 3, ss);
	cc = 0;
	for (size_t k = 0; k < nch_H; k++)
		for (size_t j = 0; j < nc_H; j++)
			for (size_t i = 0; i < nr_H; i++)
				feats_cv_temp.at<float>(i, j, k) = H[cc++];

	return feats_cv;

}

// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "hogDollarFeatL1.h"

hogDollarFeatL1::hogDollarFeatL1(bool dalalHog, int shrinkage_)
	: hogObj()
{
	if (!dalalHog) hogObj.set_params_falzen_HOG();
	featNChannels = hogObj.nchannels_hog();
	shrinkage = shrinkage_;
	hogObj.set_param_binSize(shrinkage);
}

cv::Mat hogDollarFeatL1::extract(const cv::Mat & img)
{
	cv::Mat img_temp;
	cv::cvtColor(img, img_temp, CV_RGB2BGR);
	img_temp.convertTo(img_temp, CV_32FC3);

	int nr = img.rows;
	int nc = img.cols;
	int nch = img.channels();
	int nr_H = hogObj.nrows_hog(nr);
	int nc_H = hogObj.ncols_hog(nc);
	int nch_H = hogObj.nchannels_hog();

	std::vector<float> img_vec(nr*nc*nch);
	unsigned int cc = 0;
	for (size_t k = 0; k < nch; k++)
		for (size_t j = 0; j < nc; j++)
			for (size_t i = 0; i < nr; i++)
				img_vec[cc++] = img_temp.at<cv::Vec<float, 3>>(i, j)[k];

	std::vector<float> H;
	hogObj.extract(img_vec.data(), nr, nc, nch, H);
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

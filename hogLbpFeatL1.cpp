// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "hogLbpFeatL1.h"

hogLbpFeatL1::hogLbpFeatL1()
	: hogObj()
{
	hogObj.set_params_falzen_HOG();
	featNChannels = hogObj.nchannels_hog() + lbpObj.get_num_lbpChannels();
	shrinkage = 8;
}

cv::Mat hogLbpFeatL1::extract(const cv::Mat & img)

{
	cv::Mat img_temp, img_gray, img_color;
	cv::cvtColor(img, img_temp, CV_RGB2BGR);
	img_temp.convertTo(img_color, CV_32FC3);
	cv::cvtColor(img, img_temp, CV_RGB2GRAY);
	img_temp.convertTo(img_gray, CV_32FC1);

	int nr = img.rows;
	int nc = img.cols;
	int nch = img.channels();

	std::vector<float> img_vec_color(nr*nc*nch);
	std::vector<float> img_vec_gray(nr*nc);

	int nr_hog, nc_hog, nch_hog, nr_lbp, nc_lbp, nch_lbp;
	unsigned int cc;

	cc = 0;
	for (size_t k = 0; k < nch; k++)
		for (size_t j = 0; j < nc; j++)
			for (size_t i = 0; i < nr; i++)
				img_vec_color[cc++] = img_color.at<cv::Vec<float, 3>>(i, j)[k];

	float* ptr_row;
	cc = 0;
	for (size_t i = 0; i < nr; i++)
	{
		ptr_row = img_gray.ptr<float>(i);
		for (size_t j = 0; j < nc; j++)
			img_vec_gray[cc++] = ptr_row[j];
	}

	std::vector<float> H_hog, H_lbp;

	// process HOG
	nr_hog = hogObj.nrows_hog(nr);
	nc_hog = hogObj.ncols_hog(nc);
	nch_hog = hogObj.nchannels_hog();
	hogObj.extract(img_vec_color.data(), nr, nc, nch, H_hog);

	// process LBP
	lbpObj.extract_feat(img_vec_gray.data(), nr, nc, H_lbp, nr_lbp, nc_lbp, nch_lbp, shrinkage);

	// just some checking
	if ((nr_hog != nr_lbp) || (nc_hog != nc_lbp))
	{
		printf("ERROR: no. of rows/cols of hog and lbp channel matrices do not match.\n");
		throw std::runtime_error("");
	}
	if (nch_hog + nch_lbp != featNChannels)
	{
		printf("ERROR: nch_hog + nch_lbp != featNChannels.\n");
		throw std::runtime_error("");
	}

	cv::Mat feats_cv(nr_hog, nc_hog, CV_32FC(featNChannels));

	int ss[3] = { nr_hog, nc_hog, featNChannels };
	cv::Mat feats_cv_temp = feats_cv.reshape(1, 3, ss);
	cc = 0;
	for (size_t k = 0; k < nch_hog; k++)
		for (size_t j = 0; j < nc_hog; j++)
			for (size_t i = 0; i < nr_hog; i++)
				feats_cv_temp.at<float>(i, j, k) = H_hog[cc++];
	cc = 0;
	for (size_t k = nch_hog; k < featNChannels; k++)
		for (size_t j = 0; j < nc_lbp; j++)
			for (size_t i = 0; i < nr_lbp; i++)
				feats_cv_temp.at<float>(i, j, k) = H_lbp[cc++];

	return feats_cv;
}

// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "hogVLFeatL1.h"
#include "typeExg_opencv_arma.h"

hogVLFeatL1::hogVLFeatL1(int nrows_img, int ncols_img, int nchannels_img) :
	hogObj(ncols_img, nrows_img, nchannels_img, 8, HOG_variant::HogVariantUoctti, 9, true)
{
	featNChannels = hogObj.get_num_hogChannels(); // 31
	shrinkage = 8;
}

cv::Mat hogVLFeatL1::extract(const cv::Mat & img)
{
	cv::Mat img_temp;
	cv::cvtColor(img, img_temp, CV_RGB2BGR);
	img_temp.convertTo(img_temp, CV_32FC3);
	arma::Cube<float> img_arma;
	opencv2arma<float, 3>(img_temp, img_arma);
	cv::Mat feats_cv;
	vl_hog_w hogObj2(img_arma.n_cols, img_arma.n_rows, img_arma.n_slices);
	arma2opencv<float, 31>(hogObj2.extract_feat(img_arma), feats_cv);
	return feats_cv;
}

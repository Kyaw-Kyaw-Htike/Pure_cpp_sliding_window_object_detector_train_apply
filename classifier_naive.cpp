// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "classifier_naive.h"

classifier_naive::classifier_naive()
{

}

float classifier_naive::classify(const cv::Mat & featVec)
{
	return 1;
}

void classifier_naive::train(const cv::Mat & featMatrix, const cv::Mat & labels)
{
	return;
}

void classifier_naive::save(const std::string fpath_classifier_save)
{
}

void classifier_naive::load(const std::string fpath_classifier_saved)
{
}

float classifier_naive::get_natural_thresh()
{
	return 0.0f;
}

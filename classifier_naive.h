// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "classifier_Base.h"

class classifier_naive : public classifier_Base
{
public:
	classifier_naive();
	float classify(const cv::Mat &featVec) override;
	void train(const cv::Mat &featMatrix, const cv::Mat &labels) override;
	void save(const std::string fpath_classifier_save) override;
	void load(const std::string fpath_classifier_saved) override;
	float get_natural_thresh() override;
};
// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "classifier_Base.h"

class classifier_SVM_vlfeat : public classifier_Base
{
public:
	classifier_SVM_vlfeat();
	// input should be a row vector of type CV_32FC1.
	// output is a float number.
	float classify(const cv::Mat &featVec) override;

	// load classifier model from file
	void load(const std::string &fpath);

	// train classifier
	void train(const cv::Mat &featMatrix_, const cv::Mat &labels_) override;

	float get_natural_thresh() override;

	void save(const std::string fpath_classifier_save) override;

	void load(const std::string fpath_classifier_saved) override;

protected:
	cv::Mat w_lin_;
	float bias_;
};
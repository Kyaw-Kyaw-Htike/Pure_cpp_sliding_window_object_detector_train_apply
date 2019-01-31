// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "classifier_Base.h"

class classifier_perceptron : public classifier_Base
{
public:
	classifier_perceptron();
	float classify(const cv::Mat &featVec) override;
	void train(const cv::Mat &featMatrix, const cv::Mat &labels) override;
	void save(const std::string fpath_classifier_save) override;
	void load(const std::string fpath_classifier_saved) override;
	float get_natural_thresh() override;
	void set_nepochs(int nepochs);
	void set_train_ratio(double train_ratio);
	void set_ntimes_tolerate_non_improving_val_acc(int ntimes_tolerate_non_improving_val_acc);
protected:
	cv::Mat w_lin_;
	float bias_;
	int nepochs_;
	double train_ratio_;
	int ntimes_tolerate_non_improving_val_acc_;
};

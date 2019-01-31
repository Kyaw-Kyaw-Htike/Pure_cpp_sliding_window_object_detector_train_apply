// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "classifier_Base.h"

class classifier_RF_opencv : public classifier_Base
{
public:
	classifier_RF_opencv();
	// input should be a row vector of type CV_32FC1.
	// output is a float number.
	float classify(const cv::Mat &featVec) override;

	// train classifier
	void train(const cv::Mat &featMatrix, const cv::Mat &labels_) override;

	float get_natural_thresh() override;

	void save(const std::string fpath_classifier_save) override;

	void load(const std::string fpath_classifier_saved) override;

	void set_ntrees(int ntrees_);
	void set_maxDepth(int maxDepth_);
	void set_numFeatsToSample(int numFeatsToSample_);
	void set_minSampleCount(int minSampleCount_);
		
protected:
	int ntrees;
	int maxDepth;
	int numFeatsToSample;
	int minSampleCount;
	cv::Ptr<cv::ml::RTrees> rf_obj;
};
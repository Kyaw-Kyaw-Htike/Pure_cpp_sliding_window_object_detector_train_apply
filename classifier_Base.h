// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "opencv2/opencv.hpp"

/*
Classifies an input feature vector to give a score which denotes
the confidence of the classifier. The higher the score,
the more likely the sample belongs to the positive class.
*/
class classifier_Base
{
public:
	virtual ~classifier_Base();
	// input should be a row vector of type CV_32FC1.
	// output is a float number.
	virtual float classify(const cv::Mat &featVec) = 0;
	// train classifier
	virtual void train(const cv::Mat &featMatrix, const cv::Mat &labels) = 0;
	// save the trained classifier
	virtual void save(const std::string fpath_classifier_save) = 0;
	// load the trained classifier
	virtual void load(const std::string fpath_classifier_saved) = 0;
	// return the natural threshold thresh, i.e. if the classification score > thresh, then +ve class, else background (-ve) class
	virtual float get_natural_thresh() = 0;
};

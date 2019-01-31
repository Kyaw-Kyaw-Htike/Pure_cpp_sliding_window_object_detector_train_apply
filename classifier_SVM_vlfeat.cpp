// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "classifier_SVM_vlfeat.h"
#include "vl_feat_wrappers.h"

using namespace std;

classifier_SVM_vlfeat::classifier_SVM_vlfeat()
{

}

float classifier_SVM_vlfeat::classify(const cv::Mat & featVec)
{
	return w_lin_.dot(featVec) + bias_;
}

void classifier_SVM_vlfeat::load(const std::string & fpath)
{
	cv::FileStorage fs(fpath, cv::FileStorage::READ);
	cv::Mat w;
	fs["w"] >> w; // assumes a column vector
	w_lin_ = w(cv::Range(0, w.rows - 1), cv::Range(0, 1)).t();
	w_lin_ = w_lin_.clone();
	bias_ = w.at<float>(w.rows - 1, 0);
	//cout << w_lin_ << endl;
	//cout << bias_ << endl;
	//cout << w_lin_.rows << " " << w_lin_.cols << endl;

	// to change the matlab's col majoring w to row major
	cv::Mat w_lin_CM(16, 8, CV_32FC(36));
	float *ptr = w_lin_.ptr<float>(0);
	int cc = 0;
	for (size_t k = 0; k < 36; k++)
		for (size_t j = 0; j < 8; j++)
			for (size_t i = 0; i < 16; i++)
				w_lin_CM.at<cv::Vec<float, 36>>(i, j)[k] = ptr[cc++];

	w_lin_ = w_lin_CM.reshape(1, 1).clone();

}

void classifier_SVM_vlfeat::train(const cv::Mat & featMatrix_, const cv::Mat & labels_)
{
	vl_svm_w svmObj;

	cv::Mat featMatrix;
	featMatrix_.convertTo(featMatrix, CV_64F);
	featMatrix = featMatrix.clone();

	cv::Mat labels;
	labels_.convertTo(labels, CV_64F);
	labels = labels.clone();

	printf("Training SVM classifier...\n");
	cout << "featMatrix: " << featMatrix.rows << " " << featMatrix.cols << " " << featMatrix.channels() << endl;
	cout << "labels: " << labels.rows << " " << labels.cols << " " << labels.channels() << endl;

	std::vector<double> lin_model = svmObj.train(featMatrix.ptr<double>(0), labels.ptr<double>(0),
		featMatrix.cols, featMatrix.rows, 0.001, true, true);

	printf("Classifier training completed.\n");
	w_lin_.create(1, lin_model.size() - 1, CV_32FC1);
	std::vector<float> lin_model_float(lin_model.begin(), lin_model.end());
	std::copy(lin_model_float.begin(), lin_model_float.end() - 1, w_lin_.ptr<float>(0));
	bias_ = lin_model_float[lin_model.size() - 1];
}

float classifier_SVM_vlfeat::get_natural_thresh()
{
	return 0.0f;
}

void classifier_SVM_vlfeat::save(const std::string fpath_classifier_save)
{
	cv::FileStorage fs(fpath_classifier_save, cv::FileStorage::WRITE);
	fs << "w_lin" << w_lin_ << "bias" << bias_;
	fs.release();
}

void classifier_SVM_vlfeat::load(const std::string fpath_classifier_saved)
{
	cv::FileStorage fs(fpath_classifier_saved, cv::FileStorage::READ);
	fs["w_lin"] >> w_lin_;
	fs["bias"] >> bias_;
	fs.release();
}
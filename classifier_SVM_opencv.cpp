// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "classifier_SVM_opencv.h"
#include "matrix_class_KKH.h"

using namespace std;
using namespace cv::ml;

classifier_SVM_opencv::classifier_SVM_opencv()
{

}

float classifier_SVM_opencv::classify(const cv::Mat & featVec)
{
	return w_lin_.dot(featVec) + bias_;
}

void classifier_SVM_opencv::train(const cv::Mat & featMatrix, const cv::Mat & labels_)
{
	cv::Mat labels; labels_.convertTo(labels, CV_32SC1);
	Veck<int> mk(labels.total(), labels.ptr<int>(0), false);
	// note: due to the fact that opencv treats label of -1 as the positive class, I need to
	// reverse it
	float classWeights[2] = { (mk == 1).size() / float(labels.total()), (mk == -1).size() / float(labels.total()) };
	cout << "class 0 weights = " << classWeights[0] << "; class 1 weights = " << classWeights[1] << endl;
	cv::Ptr<SVM> svm_obj = SVM::create();
	svm_obj->setType(SVM::Types::C_SVC);
	svm_obj->setC(1);
	svm_obj->setKernel(SVM::KernelTypes::LINEAR);
	svm_obj->setClassWeights(cv::Mat(1, 2, CV_32FC1, classWeights));

	cout << "featMatrix: " << featMatrix.rows << " " << featMatrix.cols << " " << featMatrix.channels() << endl;
	cout << "labels: " << labels.rows << " " << labels.cols << " " << labels.channels() << endl;

	//cv::Ptr<TrainData> train_data = TrainData::create(featMatrix, ROW_SAMPLE, labels);

	cout << "Training opencv SVM classifier...\n" << endl;

	//svm_obj->train(train_data);

	int kFold = 3;
	cv::Ptr<cv::ml::ParamGrid> CGrid = cv::ml::SVM::getDefaultGridPtr(SVM::C);
	// cv::Ptr<cv::ml::ParamGrid> CGrid = cv::ml::SVM::getDefaultGridPtr(0.0078125, 12455, 5);
	cv::Ptr<cv::ml::ParamGrid> noGrid = cv::ml::ParamGrid::create(0, 0, 0);
	svm_obj->trainAuto(featMatrix, ROW_SAMPLE, labels, kFold, CGrid, noGrid, noGrid, noGrid, noGrid, noGrid, true);
	printf("Best C value found = %f\n", svm_obj->getC());

	//svm_obj->trainAuto(train_data, 10, 
	//	SVM::getDefaultGrid(SVM::C), SVM::getDefaultGrid(SVM::GAMMA),
	//	SVM::getDefaultGrid(SVM::P), SVM::getDefaultGrid(SVM::NU),
	//	SVM::getDefaultGrid(SVM::COEF), SVM::getDefaultGrid(SVM::DEGREE), true);

	cout << "Getting decision function...\n" << endl;
	cv::Mat alpha, svidx;
	double rho = svm_obj->getDecisionFunction(0, alpha, svidx);
	cv::Mat w_dec; w_dec = svm_obj->getSupportVectors(); // only one compressed vector
	w_lin_ = w_dec.reshape(1, 1).clone();
	bias_ = -rho;

	// to account for the fact that opencv treats labels with -1 as the positive class and
	// 1 as the negative class. After doing the following, during prediction, I can then
	// correctly perform dot product and add bias_ and if that result is > 0, then 
	// positive class, i.e. label = 1
	bias_ = -bias_;
	w_lin_ = -w_lin_;
}

float classifier_SVM_opencv::get_natural_thresh()
{
	return 0.0f;
}


void classifier_SVM_opencv::save(const std::string fpath_classifier_save)
{
	cv::FileStorage fs(fpath_classifier_save, cv::FileStorage::WRITE);
	fs << "w_lin" << w_lin_ << "bias" << bias_;
	fs.release();
}

void classifier_SVM_opencv::load(const std::string fpath_classifier_saved)
{
	cv::FileStorage fs(fpath_classifier_saved, cv::FileStorage::READ);
	fs["w_lin"] >> w_lin_;
	fs["bias"] >> bias_;
	fs.release();
}

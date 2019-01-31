// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "classifier_RF_opencv.h"
#include "matrix_class_KKH.h"

using namespace std;

classifier_RF_opencv::classifier_RF_opencv()
{
	ntrees = 100;
	maxDepth = 10000;
	numFeatsToSample = -1; // sqrt(# features)
	minSampleCount = 1;
}

float classifier_RF_opencv::classify(const cv::Mat & featVec)
{
	float score = rf_obj->predict(featVec, cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
	return score/(float)ntrees; // the score will always be between -1 and 1 with the natural threshold of 0
}

void classifier_RF_opencv::train(const cv::Mat & featMatrix, const cv::Mat & labels_)
{
	cv::Mat labels; labels_.convertTo(labels, CV_32SC1);
	Veck<int> mk(labels.total(), labels.ptr<int>(0), false);
	float classWeights[2] = { (mk == 1).size() / float(labels.total()), (mk == -1).size() / float(labels.total()) };
	cout << "Pos class weight = " << classWeights[1] << "; Negative class weight = " << classWeights[0] << endl;
	rf_obj = cv::ml::RTrees::create();
	if (numFeatsToSample == -1)
		rf_obj->setActiveVarCount(std::round(std::sqrt(featMatrix.cols)));
	else
		rf_obj->setActiveVarCount(numFeatsToSample);
	rf_obj->setMaxDepth(maxDepth);
	rf_obj->setMinSampleCount(minSampleCount);
	//rf_obj->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, ntrees, 0.00001));
	rf_obj->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, ntrees, 0.00001));
	rf_obj->setPriors(cv::Mat(1, 2, CV_32FC1, classWeights));
	cout << "featMatrix: " << featMatrix.rows << " " << featMatrix.cols << " " << featMatrix.channels() << endl;
	cout << "labels: " << labels.rows << " " << labels.cols << " " << labels.channels() << endl;

	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(featMatrix, cv::ml::ROW_SAMPLE, labels);

	cout << "Training opencv RF classifier...\n" << endl;

	rf_obj->train(train_data);

	cout << "Finished training opencv RF classifier.\n" << endl;
}

float classifier_RF_opencv::get_natural_thresh()
{
	return 0.0f;
}

void classifier_RF_opencv::save(const std::string fpath_classifier_save)
{
	//cv::FileStorage fs(fpath_classifier_save, cv::FileStorage::WRITE);
	//rf_obj->write(fs);
	//fs.release();
	rf_obj->save(fpath_classifier_save);	
}

void classifier_RF_opencv::load(const std::string fpath_classifier_saved)
{
	//cv::FileStorage fs(fpath_classifier_saved, cv::FileStorage::READ);
	//fs.release();
	rf_obj = cv::ml::RTrees::load(fpath_classifier_saved);
}

void classifier_RF_opencv::set_ntrees(int ntrees_)
{
	ntrees = ntrees_;
}

void classifier_RF_opencv::set_maxDepth(int maxDepth_)
{
	maxDepth = maxDepth_;
}

void classifier_RF_opencv::set_numFeatsToSample(int numFeatsToSample_)
{
	numFeatsToSample = numFeatsToSample_;
}

void classifier_RF_opencv::set_minSampleCount(int minSampleCount_)
{
	minSampleCount = minSampleCount_;
}

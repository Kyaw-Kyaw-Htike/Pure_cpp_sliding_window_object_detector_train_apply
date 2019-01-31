// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "classifier_perceptron.h"
#include "classifier_train_utils.h"

using namespace std;



////////////////////////////////////////////////////////////////
///////////////////////// Class member function definitions ///////
////////////////////////////////////////////////////////////////


classifier_perceptron::classifier_perceptron()
{
	nepochs_ = 1000;
	train_ratio_ = 0.7;
	ntimes_tolerate_non_improving_val_acc_ = 5;
}

float classifier_perceptron::classify(const cv::Mat & featVec)
{
	return w_lin_.dot(featVec) + bias_;
}

void classifier_perceptron::train(const cv::Mat & featMatrix, const cv::Mat & labels)
{
	std::vector<int> labels_v(featMatrix.rows);
	std::copy(labels.ptr<int>(0), labels.ptr<int>(0) + featMatrix.rows, labels_v.begin());

	std::vector<int> idx_pos = stdvector_get_indices_specified_val(labels_v, 1);
	std::vector<int> idx_neg = stdvector_get_indices_specified_val(labels_v, -1);
	int npos = idx_pos.size();
	int nneg = idx_neg.size();

	printf("# train (before splitting into train and val): pos = [%d], neg = [%d]\n", npos, nneg);

	std::random_shuffle(idx_pos.begin(), idx_pos.end());
	std::random_shuffle(idx_neg.begin(), idx_neg.end());

	double val_ratio = 1 - train_ratio_;
	int npos_val = std::round(npos * val_ratio);
	int nneg_val = std::round(nneg * val_ratio);
	std::vector<int> idx_pos_val = stdvector_slice(idx_pos, 0, npos_val - 1);
	std::vector<int> idx_neg_val = stdvector_slice(idx_neg, 0, nneg_val - 1);
	idx_pos = stdvector_slice(idx_pos, npos_val, -1);
	idx_neg = stdvector_slice(idx_neg, nneg_val, -1);

	npos = npos - npos_val;
	nneg = nneg - nneg_val;

	printf("# train: pos = [%d], neg = [%d]\n", npos, nneg);
	printf("# val: pos = [%d], neg = [%d]\n", npos_val, nneg_val);

	int niters = std::max(npos, nneg);

	// initialize linear weights and bias_
	w_lin_ = cv::Mat::zeros(1, featMatrix.cols, CV_32FC1);
	bias_ = 0;

	int cc_pos = 0;
	int cc_neg = 0;
	bool pick_pos = true; // to alternate positive and negative data points
	int idx_picked;
	cv::Mat featVec;
	float label_groundtruth;

	double acc_val_best = 0;
	int count_ntimes_non_improving_val_acc = 0;
	cv::Mat w_lin_best_sofar = cv::Mat::zeros(1, featMatrix.cols, CV_32FC1);
	float bias_best_sofar = 0;
	
	for (size_t i = 0; i < nepochs_; i++)
	{
		int nwrongs = 0; // keep track of how many errors made in coming epoch
		cout << "Epoch = " << i << endl;
		for (size_t j = 0; j < niters; j++)
		{
			//cout << "Epoch " << i << ", Iter " << j << endl;
			// turn to pick a positive
			if (pick_pos)
			{
				//cout << "Turn to pick pos sample" << endl;
				// if have gone through all +ve examples, need to begin from the start
				// but after random shuffling
				if (cc_pos == npos)
				{
					//cout << "Entire pos set gone through. Restarting from beginning." << endl;
					cc_pos = 0;
					std::random_shuffle(idx_pos.begin(), idx_pos.end());
				}
				idx_picked = idx_pos[cc_pos];
				cc_pos++;
				pick_pos = false;
			}
			// turn to pick a positive
			else
			{
				//cout << "Turn to pick neg sample" << endl;
				// if have gone through all -ve examples, need to begin from the start
				// but after random shuffling
				if (cc_neg == nneg)
				{
					//cout << "Entire neg set gone through. Restarting from beginning." << endl;
					cc_neg = 0;
					std::random_shuffle(idx_neg.begin(), idx_neg.end());
				}
				idx_picked = idx_neg[cc_neg];
				cc_neg++;
				pick_pos = true;
			}

			featVec = featMatrix.row(idx_picked);
			label_groundtruth = static_cast<float>(labels.at<int>(idx_picked, 0));

			//cout << "Picking training data num " << idx_picked << " which has label " << label_groundtruth << endl;

			// if wrong prediction, then update weight
			if (classify(featVec) * label_groundtruth <= 0)
			{
				//cout << "Wrong classifier. Updating weights" << endl;
				w_lin_ += (label_groundtruth * featVec);
				bias_ += (label_groundtruth * 1.0);
				nwrongs++;
			}

		} //end j (iter)

		int num_correct = niters - nwrongs;
		double acc_train = static_cast<double>(num_correct) / niters;
		std::cout << "Training accuracy after this epoch = " << acc_train * 100 << "%" << std::endl;
		
		// compute validation accuracy
		double acc_pos_val = 0;
		double acc_neg_val = 0;

		for (size_t i = 0; i < npos_val; i++)
		{
			cv::Mat featVec_val = featMatrix.row(idx_pos_val[i]);
			float label_groundtruth_val = static_cast<float>(labels.at<int>(idx_pos_val[i], 0));
			if (classify(featVec_val) * label_groundtruth_val <= 0) // wrong
				continue;
			acc_pos_val++;
		}
		acc_pos_val = acc_pos_val / npos_val;

		for (size_t i = 0; i < nneg_val; i++)
		{
			cv::Mat featVec_val = featMatrix.row(idx_neg_val[i]);
			float label_groundtruth_val = static_cast<float>(labels.at<int>(idx_neg_val[i], 0));
			if (classify(featVec_val) * label_groundtruth_val <= 0) // wrong
				continue;
			acc_neg_val++;
		}
		acc_neg_val = acc_neg_val / nneg_val;

		// compute balanced accuracy (since the number of +ves and -ves may be different)
		double acc_val = (acc_pos_val + acc_neg_val) / 2;

		std::cout << "Validation accuracy after this epoch = " << acc_val * 100 << "%" << std::endl;

		if (acc_val <= acc_val_best)
		{
			count_ntimes_non_improving_val_acc++;
			printf("Recorded non-improving validation accuracy [current ntimes = %d]\n", count_ntimes_non_improving_val_acc);
		}
		else
		{
			w_lin_best_sofar = w_lin_.clone();
			bias_best_sofar = bias_;
			acc_val_best = acc_val;
		}
		if (count_ntimes_non_improving_val_acc > ntimes_tolerate_non_improving_val_acc_)
		{
			printf("Stopping: Validation accuracy has not improved for more than %d times.\n", ntimes_tolerate_non_improving_val_acc_);
			break;
		}

	} // end i (epoch)

	w_lin_ = w_lin_best_sofar;
	bias_ = bias_best_sofar;
}


void classifier_perceptron::save(const std::string fpath_classifier_save)
{
	cv::FileStorage fs(fpath_classifier_save, cv::FileStorage::WRITE);
	fs << "w_lin" << w_lin_ << "bias" << bias_;
	fs.release();
}

void classifier_perceptron::load(const std::string fpath_classifier_saved)
{
	cv::FileStorage fs(fpath_classifier_saved, cv::FileStorage::READ);
	fs["w_lin"] >> w_lin_;
	fs["bias"] >> bias_;
	fs.release();
}

float classifier_perceptron::get_natural_thresh()
{
	return 0.0f;
}

void classifier_perceptron::set_nepochs(int nepochs)
{
	this->nepochs_ = nepochs;
}

void classifier_perceptron::set_train_ratio(double train_ratio)
{
	train_ratio_ = train_ratio;
}

void classifier_perceptron::set_ntimes_tolerate_non_improving_val_acc(int ntimes_tolerate_non_improving_val_acc)
{
	ntimes_tolerate_non_improving_val_acc_ = ntimes_tolerate_non_improving_val_acc;
}

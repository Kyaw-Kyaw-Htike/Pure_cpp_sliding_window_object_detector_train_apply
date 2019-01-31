// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "hogDollarFeatL1.h"
#include "featL2_naive.h"
#include "classifier_perceptron.h"
#include "classifier_SVM_opencv.h"
#include "classifier_RF_opencv.h"
#include "NMSGreedy.h"
#include "slidewin_detector.h"
#include "timer_ticToc.h"

int main(int argc, char* argv[])
{
	//featL1_naive featL1_obj;
	//hogVLFeatL1 featL1_obj(128, 64, 3);
	//lbpVLFeatL1 featL1_obj(false);
	//hogLbpFeatL1 featL1_obj;
	//hogDollarFeatL1 featL1_obj(false);
	hogDollarFeatL1 featL1_obj(false, 8);
	//hogDollarFeatL1 featL1_obj(false, 4);
	//featL2_naive featL2_obj(16*8*featL1_obj.get_nchannels());
	//featL2_naive featL2_obj(16 * 8 * 31);
	//featL2_naive featL2_obj(16 * 8 * featL1_obj.get_nchannels());
	featL2_naive featL2_obj(10 * 10 * featL1_obj.get_nchannels());
	//featL2_naive featL2_obj(4 * 4 * featL1_obj.get_nchannels());
	//featL2_naive featL2_obj(128*64);

	//classifier_naive classifier_obj;
	//classifier_SVM_vlfeat classifier_obj;
	//classifier_perceptron classifier_obj;
	classifier_SVM_opencv classifier_obj;
	//classifier_RF_opencv classifier_obj;
	//classifier_obj.set_ntrees(100);
	//classifier_obj.set_train_ratio(0.6);
	//classifier_SVM_opencv classifier_obj;
	//NMS_naive nms_obj;
	NMSGreedy nms_obj;
	//NMSOpencv nms_obj;

	//slidewin_detector s(featL1_obj, featL2_obj, classifier_obj, nms_obj, 128, 64, 8);
	slidewin_detector s(featL1_obj, featL2_obj, classifier_obj, nms_obj, 80, 80, 8);
	//slidewin_detector s(featL1_obj, featL2_obj, classifier_obj, nms_obj, 40, 40, 4);

	// INRIA 
	//std::string dir_pos = "D:/Research/Datasets/INRIAPerson_Piotr/Train/imgs_crop_context/";
	//std::string dir_pos = "D:/Research/Datasets/INRIAPerson_Piotr/Train/imgs_crop_context_flip/";
	//std::string dir_pos = "C:/Users/Kyaw/Desktop/test/positive_samples_near_topdown_head/";
	std::string dir_pos = "C:/Users/Kyaw/Desktop/test/positive_samples_with_flipped_near_topdown_head/";
	//std::string dir_pos = "C:/Users/Kyaw/Desktop/test/positive_samples_40x40_with_flipped_near_topdown/";
	std::string dir_neg = "D:/Research/Datasets/INRIAPerson_Piotr/Train/images/set00/V001/";
	//std::string fpath_save_train_data = "C:/Users/Kyaw/Desktop/slidewin_detector_training_models/INRIA/trainData_64x128patches_HOG.xml";
	//std::string fpath_save_classifier = "C:/Users/Kyaw/Desktop/slidewin_detector_training_models/INRIA/classifier.xml";
	std::string fpath_save_train_data = "C:/Users/Kyaw/Desktop/slidewin_detector_training_models/INRIA/trainData_80x80Headpatches_HOG.xml";
	std::string fpath_save_classifier = "C:/Users/Kyaw/Desktop/slidewin_detector_training_models/INRIA/classifier_head.xml";
	//std::string fpath_save_train_data = "C:/Users/Kyaw/Desktop/slidewin_detector_training_models/INRIA/trainData_40x40Headpatches_HOG.xml";
	//std::string fpath_save_classifier = "C:/Users/Kyaw/Desktop/slidewin_detector_training_models/INRIA/classifier_head_40x40.xml";
		
	//// Face
	//std::string dir_pos = "D:/Research/Datasets/INRIAPerson_Piotr/Train/imgs_crop_context/";
	//std::string dir_neg = "D:/Research/Datasets/INRIAPerson_Piotr/Train/images/set00/V001/";
	//std::string fpath_save_train_data = "C:/Users/Kyaw/Desktop/train_imgs_pos/trainData_16x16patches_HOG.xml";
	//std::string fpath_save_classifier = "C:/Users/Kyaw/Desktop/train_imgs_pos/classifier.xml";
	
	s.train(dir_pos, dir_neg, true, fpath_save_train_data, true, fpath_save_classifier);
	//s.train(fpath_save_train_data, true, fpath_save_classifier);

	//classifier_obj.load(fpath_save_classifier);
	
	//cv::Mat img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00000.png");
	//cv::Mat img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00001.png");
	//cv::Mat img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00022.png");
	//cv::Mat img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00056.png");
	//cv::Mat img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00129.png");
	
	cv::Mat img = cv::imread("C:/Users/Kyaw/Desktop/test/test_set/img_1226.png");

	double scale_factor = 1;
	double thresh_visualize = -10000;

	cv::resize(img, img, cv::Size(), scale_factor, scale_factor);
	
	std::vector<cv::Rect> dr; 
	std::vector<float> ds;

	timer_ticToc tt;
	tt.tic();
	for(int i=0; i<1; i++)
	s.detect(img, dr, ds, true);
	double ttt = tt.toc();
	std::cout << "time elapsed (seconds) = " << ttt/1 << std::endl;
		
	for (size_t i = 0; i < dr.size(); i++)
	{
		if (ds[i] > thresh_visualize)
		{
			std::cout << ds[i] << std::endl;
			cv::rectangle(img, dr[i], cv::Scalar(255, 0, 0, 0), 2);
		}
	}
	cv::resize(img, img, cv::Size(), 1.0/ scale_factor, 1.0/ scale_factor);
	imshow("win", img);
	cv::waitKey(0);
	//
	//img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Test/images/set01/V000/I00001.png");
	//s.detect(img, dr, ds, true);
	//for (size_t i = 0; i < dr.size(); i++)
	//{
	//	if (ds[i] > 0)
	//	{
	//		cv::rectangle(img, dr[i], cv::Scalar(255, 0, 0, 0), 2);
	//		//cout << ds[i] << endl;
	//	}

	//}
	//imshow("win", img);
	//cv::waitKey(0);




	//cout << m2.rows << " " << m2.cols << " " << m2.channels() << endl;

	//timer_ticToc tt;
	//tt.tic();	
	////cv::Mat feats_sampled = s.get_feats_img(img, 100);
	//img = cv::imread("D:/Research/Datasets/INRIAPerson_Piotr/Train/imgs_crop_context/INRIA_pos_00001.png");
	//std::vector<cv::Rect> dr_sampled = s.get_dr_img(img, -1);
	//cv::Mat feats_sampled = s.get_feats_img(img, -1);
	//cout << "Time taken = " << tt.toc() << " secs" << endl;

	//for (size_t i = 0; i < dr_sampled.size(); i++)
	//{
	//	cv::Mat img2 = img.clone();
	//	cv::rectangle(img2, dr_sampled[i], cv::Scalar(255, 0, 0, 0), 2);
	//	imshow("win", img2);
	//	cv::waitKey(0);
	//}

	//cout << "feats_sampled " << feats_sampled.rows << " " << feats_sampled.cols << endl;
	//cout << feats_sampled << endl;
	//for (size_t i = 0; i < feats_sampled.rows; i++)
	//{
	//	cv::Mat temp;
	//	feats_sampled.row(i).reshape(1, 128).convertTo(temp, CV_8UC1);
	//	imshow("win", temp);
	//	cv::waitKey(0);
	//}

	//cv::imshow("win", img); cv::waitKey(0);
	//cv::Rect roi(10, 10, 64, 128);
	//cv::Mat img2 = cv::Mat(img, roi);
	//cout << img2.rows << " " << img2.cols << endl;
	//cv::imshow("win", img2); cv::waitKey(0);

	//slidewin_detector_cv s;

	//slidewin_detector s;
	//s.strideX = 8;
	//s.strideY = 8;

	//timer_ticToc tt;
	//tt.tic();
	//for (size_t i = 0; i < 1; i++)
	//{
	//	s.extract_patches(img);
	//}	
	//double time_taken = tt.toc();
	//cout << "Time for 1 frames = " << time_taken << " secs" << endl;
	//cout << "Time for 30 frame = " << time_taken * 30 << " secs" << endl;	


	return 0;
}
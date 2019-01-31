// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include "featL1_Base.h"
#include "featL2_Base.h"
#include "classifier_Base.h"
#include "NMS_Base.h"

/*
This class trains an object detector to detect objects using multi-scale sliding window scheme.
The feature extraction, classifier and NMS components are generic and can be anything as long
as they inherit from the abstract/base classes featL1_Base, featL2_Base, classifier_Base and
NMS_Base and they override the respective methods.
Inputs to the training process are (1) directory to cropped positive patches
(all same size as detection window size)
and (2) directory to negative images from where negatives and hard negatives will be data mined.
After the training process, I can run the detector on an image of any size to detect the object
category.
*/
class slidewin_detector
{
private:

	// ===================================
	// Objects for feature extraction, classification and NMS
	// ===================================
	featL1_Base &featL1_obj;
	featL2_Base &featL2_obj;
	classifier_Base &classifier_obj;
	NMS_Base &NMS_Obj;

	// ===================================
	// for original image: the following params can be set by the user
	// in the constructor
	// ===================================
	// size of detection window ([0]: num rows; [1]: num cols)
	int winsize[2];
	// stride; same size for both horizontal and vertical slides
	int stride;
	// the ratio between two scales for sliding window. 
	// the smaller, the finer the scales and thus, the more scales 
	// needed to processed
	double scaleratio;
	// in case user wants to limit the maximum number of scales
	// e.g. for very small objects in very large images.
	// normally, just set this number to a very large number (inf).
	int max_nscales; // if want to limit number of scales

					 // ===================================
					 // for feat channel "image": automatically computed params
					 // during the constructor
					 // ===================================
					 // how much shrinking does the L1 feature extraction perform
	int shrinkage_channel;
	// the stride on the channel "image". This should map back to the 
	// stride on the original image.
	int stride_channel;
	// size of window on the channel "image". This should map back to the 
	// winsize on the original image.
	int winsize_channel[2];
	// number of channels of the channel "image".
	int nchannels_channel;
	int ndims_feat; // length of the feature vector after L2 feature extraction

					// ===================================
					// output data of the private "process_img" method; these data correponds to
					// processing of a particular image size. These data members are written 
					// after the "process_img" method is called. The process_img is a private method
					// that does multi-scale sliding window processing including feature extraction,
					// classification, etc. The user can call public methods which make use of this
					// "process_img" method to access important functionalities.
					// ===================================
					// store info about for what image size the following "parameters" has been
					// prepared for.
	int nrows_img_prep, ncols_img_prep;
	// no. of scales that image sliding window must process
	int num_scales;
	// vector of scales computed for sliding window; scales.size()==num_scales
	std::vector<double> scales;
	// total no. of sliding windows for the image (across all the scales).
	unsigned int nslidewins_total;
	// vector of sliding window rectangles. dr.size()==nslidewins_total
	std::vector<cv::Rect> dr;
	// for each sliding window rectangle, which scale did it come from;
	// stores the index to std::vector<double>scales
	std::vector<unsigned int> idx2scale4dr;
	// vector of sliding window classification scores. ds.size()==nslidewins_total
	// ds will only be written if det_mode (which is one of the arguments to the
	// process_img method) is true.
	std::vector<float> ds;
	// matrix of features for randomly sampled sliding windows.
	// this will only be written if det_mode = false;
	cv::Mat feats_slidewin;

	// ===================================
	// Other data
	// ===================================
	// directory of cropped image patches for positive class for training classifier
	std::string dir_pos;
	// directory of full negative images for training classifier
	std::string dir_neg;
	// the file path to training data feature matrix and labels for training classifier
	std::string traindata_fpath_save;

	// ===================================
	// Private member functions
	// ===================================

	void check_params_constructor();

	// a convenient method for clearing all data in std::vector.
	// both vector size and capacity becomes zero.
	template <class T>
	void clean_vector(std::vector<T> &v);

	// given an image, it will process in a sliding window manner
	// and write the "outputs"/results to certain private data members.
	// In the method argument, if save_feats == true, feature matrix of
	// of the sliding window rectangles will be saved in the data member
	// "feats_slidewin", where each row is the feature vector for one
	// sliding window rectangle (after L1 + L2 feature extraction).
	// be careful however that for very large images and very high dimensional
	// features, memory requirements might be too large.
	// if apply_classifier is false, the classifier will not be applied for the
	// sliding window. This should only be used when for example only want 
	// dr and feats_slidewin (when sampling features, dr, etc.)
	// dr will always be computed and saved in all cases.
	void process_img(const cv::Mat &img, bool save_feats = false, bool apply_classifier = true);

public:

	// no default destructor allowed.
	slidewin_detector() = delete;

	// The constructor where the user need specify feature extraction, 
	// classifier and NMS objects. Default params are set for sliding window scheme.
	// if user wants to change these default sliding window params, 
	// use the method "set_params"
	slidewin_detector(featL1_Base &a, featL2_Base &b, classifier_Base &c, NMS_Base &d);

	// The constructor where the user need specify feature extraction, 
	// classifier and NMS objects, and also params for sliding window scheme.
	slidewin_detector(featL1_Base &a, featL2_Base &b, classifier_Base &c, NMS_Base &d,
		int winsize_nrows, int winsize_ncols, int stride_ = 8,
		double scaleratio_ = std::pow(2, 1 / 8.0),
		int max_nscales_ = std::numeric_limits<int>::max());


	// get feature vectors from given image from multi-scale sliding window space.
	// Useful for initially sampling negatives for training detector, etc.
	// if nsamples=-1, no sampling; return all features 
	// according to all sliding windows in order
	cv::Mat get_feats_img(const cv::Mat &img, int nsamples = -1);

	// get rectangles from given image from multi-scale sliding window space.
	// if nsamples=-1, no sampling; return all sliding windows in order
	std::vector<cv::Rect> get_dr_img(const cv::Mat &img, int nsamples = -1);

	// detect objects on the given image with the classifier
	void detect(const cv::Mat &img, std::vector<cv::Rect> &dr_, std::vector<float> &ds_, bool apply_NMS = true);

	// train by extracting features from a directory of cropped positive patches, a directory of
	// full negative images (where hard negs will be mined). Optionally, all the extracted features
	// can be saved so that later on, if desired, I can use other overloaded train function
	// which just loads the saved features and labels for training
	void train(std::string dir_pos_, std::string dir_neg_, bool save_train_feats = false, std::string traindata_fpath_save = "", bool save_classifier = false, std::string classifier_fpath_save = "");

	// train by loading feature matrix and labels from the saved file
	void train(std::string traindata_fpath_save_, bool save_classifier = false, std::string classifier_fpath_save = "");

};

template<class T>
inline void slidewin_detector::clean_vector(std::vector<T>& v)
{
	//v.clear();
	std::vector<T>().swap(v);
}


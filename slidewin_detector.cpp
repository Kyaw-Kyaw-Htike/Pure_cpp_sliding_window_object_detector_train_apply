// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#define NOMINMAX
#include "slidewin_detector.h"
#include <random>
#include <numeric>
#include "fileIO_helpers.h"

using namespace std;

void slidewin_detector::check_params_constructor()
{
	if (stride % shrinkage_channel != 0)
	{
		printf("ERROR: stride MOD shrinkage_channel != 0.\n");
		throw std::runtime_error("");
	}
	if (winsize[0] % shrinkage_channel != 0)
	{
		printf("ERROR: winsize[0] MOD shrinkage_channel != 0.\n");
		throw std::runtime_error("");
	}
	if (winsize[1] % shrinkage_channel != 0)
	{
		printf("ERROR: winsize[1] MOD shrinkage_channel != 0.\n");
		throw std::runtime_error("");
	}
}

void slidewin_detector::process_img(const cv::Mat & img, bool save_feats, bool apply_classifier)
{
	nrows_img_prep = img.rows; // to record the private data member
	ncols_img_prep = img.cols; // to record the private data member
	int nrows_img = nrows_img_prep; // for use locally in this method
	int ncols_img = ncols_img_prep; // for use locally in this method

	clean_vector(scales);
	clean_vector(dr);
	clean_vector(idx2scale4dr);
	if (apply_classifier) clean_vector(ds);

	// compute analytically how many scales there are for sliding window.
	// this formula gives the same answer as would be computed in a loop.
	num_scales = std::min(std::floor(std::log(static_cast<double>(nrows_img) / winsize[0]) / std::log(scaleratio)),
		std::floor(std::log(static_cast<double>(ncols_img) / winsize[1]) / std::log(scaleratio))) + 1;

	// preallocate for efficiency
	scales.resize(num_scales);

	// find a tight upper bound on total no. of sliding windows needed
	double stride_scale, nsw_rows, nsw_cols;
	size_t nslidewins_total_ub = 0;
	for (size_t s = 0; s < num_scales; s++)
	{
		stride_scale = stride*std::pow(scaleratio, s);
		nsw_rows = std::floor(nrows_img / stride_scale) - std::floor(winsize[0] / stride) + 1;
		nsw_cols = std::floor(ncols_img / stride_scale) - std::floor(winsize[1] / stride) + 1;
		// Without the increment below, I get exact computation of number of sliding
		// windows, but just in case (to upper bound it)
		++nsw_rows; ++nsw_cols;
		nslidewins_total_ub += (nsw_rows * nsw_cols);
	}

	//cout << "nrows_img: " << nrows_img << " " << "ncols_img: " << ncols_img << endl;
	//cout << "num_scales: " << num_scales << endl;
	//cout << "nslidewins_total_ub: " << nslidewins_total_ub << endl;
	//cout << "num channels of L1 feat channel image: " << nchannels_channel << endl;
	//cout << "ndims_feat: " << ndims_feat << endl;

	// preallocate/reserve for speed		
	dr.reserve(nslidewins_total_ub);
	idx2scale4dr.reserve(nslidewins_total_ub);
	if (apply_classifier) ds.reserve(nslidewins_total_ub);
	if (save_feats)
	{
		feats_slidewin = cv::Mat();
		feats_slidewin.reserve(nslidewins_total_ub);
	}

	// the resized image and the channel image
	cv::Mat img_cur, H, feat_vec;
	// reset counter for total number of sliding windows across all scales
	nslidewins_total = 0;

	for (size_t s = 0; s < num_scales; s++)
	{
		// compute how much I need to scale the original image for this current scale s
		scales[s] = std::pow(scaleratio, s);
		// get the resized version of the original image with the computed scale
		cv::resize(img, img_cur, cv::Size(), 1.0 / scales[s], 1.0 / scales[s], cv::INTER_LINEAR);

		// use L1 feature extractor to extract features from this resized image
		H = featL1_obj.extract(img_cur);

		// run sliding window in the channel image space 
		for (size_t i = 0; i < H.rows - winsize_channel[0] + 1; i += stride_channel)
		{
			for (size_t j = 0; j < H.cols - winsize_channel[1] + 1; j += stride_channel)
			{
				// save the current sliding window rectangle after mapping back:
				// (1) map from channel "image" space to image space (at this scale)
				// (2) map back to image space at this scale to original scale

				dr.push_back(cv::Rect(
					std::round(((j + 1)*shrinkage_channel - shrinkage_channel)*scales[s]),
					std::round(((i + 1)*shrinkage_channel - shrinkage_channel)*scales[s]),
					std::round((winsize[1]) * scales[s]),
					std::round((winsize[0]) * scales[s]))
				);

				// stores which scale of the original image this dr comes from
				idx2scale4dr.push_back(s);

				// Get the channel image patch according to this current sliding window
				// rectangle, extract L2 features which will output a feature vector.
				feat_vec = featL2_obj.extract(
					H(
						cv::Rect(j, i, winsize_channel[1], winsize_channel[0])
					)
				);

				// apply classifier on the feature vector and save it
				if (apply_classifier) ds.push_back(classifier_obj.classify(feat_vec));

				// save the extracted features
				if (save_feats) feats_slidewin.push_back(feat_vec);

				++nslidewins_total;

			} // end j
		} //end i

	} //end s

} //end method

slidewin_detector::slidewin_detector(featL1_Base & a, featL2_Base & b, classifier_Base & c, NMS_Base & d)
	:featL1_obj(a), featL2_obj(b), classifier_obj(c), NMS_Obj(d)
{
	winsize[0] = 128; // num rows of detection window size 
	winsize[1] = 64; // num cols of detection window size
	stride = 8;
	scaleratio = std::pow(2, 1 / 8.0);
	max_nscales = std::numeric_limits<int>::max();

	nrows_img_prep = 0;
	ncols_img_prep = 0;

	shrinkage_channel = featL1_obj.get_shrinkage();
	nchannels_channel = featL1_obj.get_nchannels();
	ndims_feat = featL2_obj.get_ndimsFeat();
	stride_channel = stride / shrinkage_channel;
	winsize_channel[0] = winsize[0] / shrinkage_channel;
	winsize_channel[1] = winsize[1] / shrinkage_channel;

	check_params_constructor();
}

slidewin_detector::slidewin_detector(featL1_Base & a, featL2_Base & b, classifier_Base & c, NMS_Base & d, int winsize_nrows, int winsize_ncols, int stride_, double scaleratio_, int max_nscales_)
	:featL1_obj(a), featL2_obj(b), classifier_obj(c), NMS_Obj(d)
{
	winsize[0] = winsize_nrows;
	winsize[1] = winsize_ncols;
	stride = stride_;
	scaleratio = scaleratio_;
	max_nscales = max_nscales_;

	nrows_img_prep = 0;
	ncols_img_prep = 0;

	shrinkage_channel = featL1_obj.get_shrinkage();
	nchannels_channel = featL1_obj.get_nchannels();
	ndims_feat = featL2_obj.get_ndimsFeat();
	stride_channel = stride / shrinkage_channel;
	winsize_channel[0] = winsize[0] / shrinkage_channel;
	winsize_channel[1] = winsize[1] / shrinkage_channel;

	check_params_constructor();
}

cv::Mat slidewin_detector::get_feats_img(const cv::Mat & img, int nsamples)
{
	// get all feats first
	process_img(img, true, false);

	if (nsamples < 0) return feats_slidewin;

	// prepare random number generator which will randomly
	// sample from the integer set {0,1,...,nslidewins_total-1}
	std::default_random_engine eng{ std::random_device{}() };
	std::uniform_int_distribution<int> idis(0, nslidewins_total - 1);

	cv::Mat feats_sampled(nsamples, ndims_feat, CV_32FC1);

	for (size_t i = 0; i < nsamples; i++)
		feats_slidewin.row(idis(eng)).copyTo(feats_sampled.row(i));

	return feats_sampled;
}

std::vector<cv::Rect> slidewin_detector::get_dr_img(const cv::Mat & img, int nsamples)
{
	// to get all dr first
	process_img(img, false, false);

	// if no sampling, then just return everything
	if (nsamples < 0) return dr;

	// prepare random number generator which will randomly
	// sample from the integer set {0,1,...,nslidewins_total-1}
	std::default_random_engine eng{ std::random_device{}() };
	std::uniform_int_distribution<int> idis(0, nslidewins_total - 1);

	std::vector<cv::Rect> dr_sampled(nsamples);
	for (size_t i = 0; i < nsamples; i++)
		dr_sampled[i] = dr[idis(eng)];

	return dr_sampled;
}

void slidewin_detector::detect(const cv::Mat & img, std::vector<cv::Rect>& dr_, std::vector<float>& ds_, bool apply_NMS)
{
	process_img(img, false, true);
	std::vector<cv::Rect> dr_temp;
	std::vector<float> ds_temp;
	if (apply_NMS)
	{
		std::vector<cv::Rect> dr_nms;
		std::vector<float> ds_nms;
		NMS_Obj.suppress(dr, ds, dr_nms, ds_nms);
		dr_temp = dr_nms;
		ds_temp = ds_nms;
	}
	else
	{
		dr_temp = dr;
		ds_temp = ds;
	}

	int nrects = dr_temp.size();
	dr_.clear();
	ds_.clear();
	dr_.reserve(nrects);
	ds_.reserve(nrects);

	for (size_t i = 0; i < nrects; i++)
	{
		if (ds_temp[i] > classifier_obj.get_natural_thresh())
		{
			dr_.push_back(dr_temp[i]);
			ds_.push_back(ds_temp[i]);
		}
	}
}

// note: below is for when dir_fnames() don't work
//#include <filesystem>
//std::vector<std::string> get_filenames(std::string dir_files)
//{
//	std::vector<std::string> ss;
//	for (auto & p : std::experimental::filesystem::directory_iterator(dir_files))
//	{
//		ss.push_back(std::experimental::filesystem::absolute(p.path()).generic_string());
//	}
//	return ss;
//}

void slidewin_detector::train(std::string dir_pos_, std::string dir_neg_, bool save_train_feats, std::string traindata_fpath_save, bool save_classifier, std::string classifier_fpath_save)
{
	// just for recording so that in the future, I have a record of which training data
	// the detector was trained with
	dir_pos = dir_pos_;
	dir_neg = dir_neg_;

	// read in image full path names
	std::vector<std::string> fnames_pos, fnames_neg;

	dir_fnames(dir_pos, { "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff" }, fnames_pos);
	dir_fnames(dir_neg, { "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff" }, fnames_neg);

	//// below is for when dir_fnames doesn't work
	//fnames_pos = get_filenames(dir_pos);
	//fnames_neg = get_filenames(dir_neg);

	int npos = fnames_pos.size();
	int nnegImg = fnames_neg.size();

	// read cropped patches to form positive class of the dataset
	cv::Mat feats_pos(npos, ndims_feat, CV_32FC1);
	cv::Mat img;
	printf("Extracting features from cropped +ve class...\n");
	for (size_t i = 0; i < npos; i++)
	{
		img = cv::imread(fnames_pos[i]);
		get_feats_img(img).copyTo(feats_pos.row(i));
	}
	printf("Extracting +ve features done.\n");
	cout << "feats_pos info: " << feats_pos.rows << " " << feats_pos.cols << " " << feats_pos.channels() << endl;

	// random sample negative patches and features from negative images
	int num_ini_negImg = 100;
	int num_nsamples_per_img = 100;
	vector<cv::Mat> feats_neg_ini_vec(num_ini_negImg);
	for (size_t i = 0; i < num_ini_negImg; i++)
	{
		img = cv::imread(fnames_neg[i]);
		feats_neg_ini_vec[i] = get_feats_img(img, num_nsamples_per_img);
	}
	cv::Mat feats_neg_ini;
	cv::vconcat(feats_neg_ini_vec, feats_neg_ini);
	cout << "feats_neg_ini info: " << feats_neg_ini.rows << " " << feats_neg_ini.cols << " " << feats_neg_ini.channels() << endl;

	// train classifier with current initially collected dataset
	cv::Mat labels(npos + feats_neg_ini.rows, 1, CV_32SC1);
	labels(cv::Range(0, npos), cv::Range(0, 1)).setTo(1);
	labels(cv::Range(npos, npos + feats_neg_ini.rows), cv::Range(0, 1)).setTo(-1);
	cv::Mat feats_train;
	cv::Mat feats_train_[2] = { feats_pos, feats_neg_ini };
	cv::vconcat(feats_train_, 2, feats_train);
	classifier_obj.train(feats_train, labels);

	// go through negative images to find hard negs
	cout << "Looking for hard negs...\n" << endl;
	int nhardnegs = 10000;
	std::vector<cv::Rect> dr_dets;
	std::vector<float> ds_dets;
	unsigned int nfp, tnfp;
	cv::Mat feats_neg_hard;
	cv::Mat img_roi;
	feats_neg_hard.reserve(nhardnegs);

	tnfp = 0;
	for (size_t i = 0; i < nnegImg; i++)
	{
		img = cv::imread(fnames_neg[i]);
		detect(img, dr_dets, ds_dets, true);
		nfp = dr_dets.size();

		for (size_t j = 0; j < nfp; j++)
		{
			//cv::rectangle(img, dr_dets[j], cv::Scalar(255, 0, 0, 0), 2);
			// just in case the given dr_deets[j] overshoots a bit (by 1 or 2 pixels)
			// the image boundary, in order to prevent crashing
			if (dr_dets[j].x + dr_dets[j].width >= img.cols || dr_dets[j].y + dr_dets[j].height >= img.rows)
				continue;
			img_roi = img(dr_dets[j]);
			feats_neg_hard.push_back(get_feats_img(img_roi, 1));
			tnfp++;
		}
		//cv::imshow("win", img); cv::waitKey(0);

		cout << "Number of false +ves found in image " << i << " = " << nfp << endl;
		cout << "Total Number of false +ves collected so far = " << tnfp << endl;

		if (tnfp >= nhardnegs)
		{
			cout << "Stopped due to having reached target of " << nhardnegs << " hard neg samples" << endl;
			break;
		}
	}

	// Prepare to retrain classifier
	cout << "Preparing to retrain classifier with hard negs...\n" << endl;
	feats_train_[0] = feats_train;
	feats_train_[1] = feats_neg_hard;
	cv::vconcat(feats_train_, 2, feats_train);
	cv::Mat labels_hardneg(tnfp, 1, CV_32SC1);
	labels_hardneg.setTo(-1);
	cv::Mat labels_[2] = { labels, labels_hardneg };
	cv::vconcat(labels_, 2, labels);

	// optionally, save the features (i.e. pos, neg and hard neg features & all labels)
	if (save_train_feats)
	{
		cout << "Saving feature matrix and labels...\n" << endl;
		cv::FileStorage fs(traindata_fpath_save, cv::FileStorage::WRITE);
		fs << "feats_train" << feats_train << "labels" << labels;
		fs.release();
	}

	cout << "Re-training classifier (final classifier)...\n" << endl;
	classifier_obj.train(feats_train, labels);

	// optionally, save the trained classifier
	if (save_classifier)
	{
		cout << "Saving trained classifier...\n" << endl;
		classifier_obj.save(classifier_fpath_save);
		cout << "Trained classifier saved.\n" << endl;
	}

	cout << "Sliding window detector training done.\n" << endl;
}

void slidewin_detector::train(std::string traindata_fpath_save_, bool save_classifier, std::string classifier_fpath_save)
{
	traindata_fpath_save = traindata_fpath_save_; // just record it
	cout << "Loading feature matrix and labels...\n" << endl;
	cv::FileStorage fs(traindata_fpath_save, cv::FileStorage::READ);
	cv::Mat feats_train, labels;
	fs["feats_train"] >> feats_train;
	fs["labels"] >> labels;
	fs.release();

	cout << "Training classifier...\n" << endl;
	classifier_obj.train(feats_train, labels);

	// optionally, save the trained classifier
	if (save_classifier)
	{
		cout << "Saving trained classifier...\n" << endl;
		classifier_obj.save(classifier_fpath_save);
		cout << "Trained classifier saved.\n" << endl;
	}
}

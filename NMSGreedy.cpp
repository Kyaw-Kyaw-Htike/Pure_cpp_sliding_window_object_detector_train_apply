// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#include "NMSGreedy.h"

NMSGreedy::NMSGreedy()
{
	overlap_thresh = 0.5;
}

void NMSGreedy::set_thresh(float th)
{
	overlap_thresh = th;
}

void NMSGreedy::suppress(const std::vector<cv::Rect>& dr, const std::vector<float>& ds, std::vector<cv::Rect>& dr_nms, std::vector<float>& ds_nms)
{
	arma::Mat<float> dr_(dr.size(), 4);
	arma::Col<float> ds_(dr.size());
	arma::Mat<float> dr_new;
	arma::Col<float> ds_new;

	for (size_t i = 0; i < dr.size(); i++)
	{
		dr_.at(i, 0) = dr[i].x;
		dr_.at(i, 1) = dr[i].y;
		dr_.at(i, 2) = dr[i].width;
		dr_.at(i, 3) = dr[i].height;
		ds_.at(i) = ds[i];
	}

	// process NMS
	merge_dets(dr_, ds_, dr_new, ds_new);

	int ndr_nms = dr_new.n_rows;
	dr_nms.resize(ndr_nms);
	ds_nms.resize(ndr_nms);

	for (size_t i = 0; i < ndr_nms; i++)
	{
		dr_nms[i].x = dr_new.at(i, 0);
		dr_nms[i].y = dr_new.at(i, 1);
		dr_nms[i].width = dr_new.at(i, 2);
		dr_nms[i].height = dr_new.at(i, 3);
		ds_nms[i] = ds_new.at(i);
	}
}

void NMSGreedy::merge_dets(const arma::Mat<float>& dr, const arma::Col<float>& ds, arma::Mat<float>& dr_new, arma::Col<float>& ds_new)
{
	dr_new.set_size(0, 0);
	ds_new.set_size(0);

	if (dr.n_rows == 0) return;

	arma::Col<float> x1 = dr.col(0);
	arma::Col<float> y1 = dr.col(1);
	arma::Col<float> x2 = dr.col(0) + dr.col(2);
	arma::Col<float> y2 = dr.col(1) + dr.col(3);
	arma::Col<float> s = ds;

	arma::Col<float>area = (x2 - x1 + 1) % (y2 - y1 + 1);
	arma::uvec I = arma::sort_index(s);
	arma::Col<float> vals = s(I);

	arma::uvec pick = arma::zeros<arma::uvec>(s.n_elem);
	unsigned int counter = 0;

	arma::Col<float> temp1, temp2, w, h, o, xx1, xx2, yy1, yy2;
	arma::uvec I_sub;

	while (I.n_elem > 0)
	{
		int last = I.n_elem;
		unsigned int i = I(last - 1);
		pick(counter) = i;
		counter++;

		if (last == 1) break;

		I_sub = I.subvec(0, last - 2);

		temp1 = x1(I_sub);
		temp2 = arma::repmat(arma::Mat<float>(&(x1(i)), 1, 1), temp1.n_elem, 1);
		xx1 = arma::max(temp2, temp1);

		temp1 = x2(I_sub);
		temp2 = arma::repmat(arma::Mat<float>(&(x2(i)), 1, 1), temp1.n_elem, 1);
		xx2 = arma::min(temp2, temp1);

		temp1 = y1(I_sub);
		temp2 = arma::repmat(arma::Mat<float>(&(y1(i)), 1, 1), temp1.n_elem, 1);
		yy1 = arma::max(temp2, temp1);

		temp1 = y2(I_sub);
		temp2 = arma::repmat(arma::Mat<float>(&(y2(i)), 1, 1), temp1.n_elem, 1);
		yy2 = arma::min(temp2, temp1);

		temp1 = xx2 - xx1 + 1;
		temp2 = arma::zeros<arma::Col<float>>(temp1.n_elem);
		w = arma::max(temp2, temp1);

		temp1 = yy2 - yy1 + 1;
		temp2 = arma::zeros<arma::Col<float>>(temp1.n_elem);
		h = arma::max(temp2, temp1);

		o = w % h / area(I_sub);
		I = I(arma::find(o <= overlap_thresh));

	}

	pick = pick.subvec(0, counter - 1);
	dr_new = dr.rows(pick);
	ds_new = ds(pick);
}

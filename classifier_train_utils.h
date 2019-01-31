// Copyright (C) 2019 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
#pragma once

#include <vector>

//template<typename T>
//std::vector<int> stdvector_get_indices_specified_val(std::vector<T> vec, T val);
//
//template<typename T>
//std::vector<T> stdvector_slice(std::vector<T> vec, int idx_start, int idx_end);


template<typename T>
std::vector<int> stdvector_get_indices_specified_val(std::vector<T> vec, T val)
{
	std::vector<int> indices;
	indices.reserve(vec.size());
	for (size_t i = 0; i < vec.size(); i++)
	{
		if (vec[i] == val) indices.push_back(i);
	}
	return indices;
}

template<typename T>
std::vector<T> stdvector_slice(std::vector<T> vec, int idx_start, int idx_end)
{
	int ndata = vec.size();
	if (idx_start == -1) idx_start = ndata - 1; // -1 is a special notation for last one
	if (idx_end == -1) idx_end = ndata - 1; // -1 is a special notation for last one

	std::vector<T> vecOut(idx_end - idx_start + 1);
	T* ptr = vec.data();
	T* ptr_out = vecOut.data();
	for (size_t i = idx_start; i <= idx_end; i++)
		ptr_out[i - idx_start] = ptr[i];
	return vecOut;
}
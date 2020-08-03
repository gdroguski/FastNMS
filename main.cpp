#include <iostream>
#include <vector>
#include "clipper.hpp"
#include <chrono>

#include <fstream>
#include <string>
#include <sstream>


#include <iomanip>

using namespace ClipperLib;

template<class T>
using vector = std::vector<T>;

bool getFileContent(std::string fileName, vector<std::string>& vecOfStrs)
{
	// Open the File
	std::ifstream in(fileName.c_str());
	// Check if object is valid
	if (!in)
	{
		std::cerr << "Cannot open the File : " << fileName << std::endl;
		return false;
	}
	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if (str.size() > 0)
			vecOfStrs.push_back(str);
	}
	//Close The File
	in.close();
	return true;
}

vector<std::string> split(const std::string& s, char delim) {
	std::stringstream ss(s);
	std::string item;
	vector<std::string> elems;
	while (std::getline(ss, item, delim))
		 elems.push_back(std::move(item)); 

	return elems;
}

double area(const float* polygon, const int& size_x, const int& size_y)
{
	float area = 0.0;
	int n = size_x * size_y;
	int prev_idx;

	float curr_pt;
	float next_pt;
	float prev_pt;

	for (int i = 1; i < size_x + 1; i++)
	{
		curr_pt = polygon[(i * size_y) % n + 0];
		next_pt = polygon[((i + 1) * size_y) % n + 1];
		prev_idx = ((size_x + (i - 1) % n) * size_y)% n;
		prev_pt = polygon[prev_idx + 1];

		area += curr_pt * (next_pt - prev_pt);
	}

	return (double)abs(area / 2);
}

std::pair<vector<int>, float*> nms(
	const vector<vector<float>>& input_boxes, 
	const double& overlapThresh, const double& neighbourThresh, 
	const float& minScore, const int& num_neig
)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	const int n = input_boxes.size();
	const int m = input_boxes[0].size(); // 8, [x1, y1, x2, y2, x3, y3, x4, y4, score]

	int arr_size = n * m;
	float* boxes = new float[arr_size];
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			boxes[i * m + j] = input_boxes[i][j];

	float* new_boxes = new float[arr_size];
	vector<int> pick;
	bool* suppressed = new bool[n] {false};

	double* areas = new double[n];
	float ref_polygon[4 * 2];
	for (int i = 0; i < n; i++) // calculate area of each quadrilateral
	{
		for (int j = 0; j < 8; j++)
			ref_polygon[j] = boxes[i * m + j];

		areas[i] = area(ref_polygon, 4, 2);
	}

	Path* polygons = new Path[n];
	Path ref_path(4);
	cInt* ref_scaled_polygon = new cInt[4 * 2];
	for (int i = 0; i < n; i++) // scale each quadrilateral to integer values
	{
		for (int j = 0; j < 8; j++)
			ref_scaled_polygon[j] = (cInt)((double)(boxes[i * m + j] * std::pow(2.0, 31)));
		for (int j = 0; j < 4; j++)
		{
			ref_path[j].X = ref_scaled_polygon[j*2+0];
			ref_path[j].Y = ref_scaled_polygon[j*2+1];
		}
		polygons[i] = ref_path;
	}

	arr_size = n * 2;
	double* centers = new double[arr_size];
	double* sides = new double[arr_size];
	cInt ref_coords_x[4];
	cInt ref_coords_y[4];
	double max_x, min_x, max_y, min_y;
	for (int i = 0; i < n; i++) // calculate centers and bboxes of quadrilaterals
	{
		ref_path = polygons[i];
		centers[i * 2 + 0] = (ref_path[0].X + ref_path[1].X + ref_path[2].X + ref_path[3].X) / 4.0;
		centers[i * 2 + 1] = (ref_path[0].Y + ref_path[1].Y + ref_path[2].Y + ref_path[3].Y) / 4.0;
		for (int j = 0; j < 4; j++)
		{
			ref_coords_x[j] = ref_path[j].X;
			ref_coords_y[j] = ref_path[j].Y;
		}
		max_x = *std::max_element(ref_coords_x, ref_coords_x + 4);
		min_x = *std::min_element(ref_coords_x, ref_coords_x + 4);
		max_y = *std::max_element(ref_coords_y, ref_coords_y + 4);
		min_y = *std::min_element(ref_coords_y, ref_coords_y + 4);

		sides[i * 2 + 0] = max_x - min_x;
		sides[i * 2 + 1] = max_y - min_y;
	}

	float* order_floats = new float[n];
	int* order_ints = new int[n];
	for(int i =0; i<n; i++) // argsort by NN score
	{
		order_floats[i] = boxes[i * m + 8];
		order_ints[i] = i;
	}
	std::sort(order_ints, order_ints + n, [&order_floats](int lhs, int rhs) { return order_floats[lhs] < order_floats[rhs]; });
	std::reverse(order_ints, order_ints + n);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time pre loop nms: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;
	begin = std::chrono::steady_clock::now();
	Clipper c;
	Paths solutions;
	arr_size = n * 8;
	float* box_to_agg = new float[arr_size];
	int* neighbours = new int[n];
	int neigh_cnt = 0;
	for (int _i = 0; _i < n; _i++)
	{
		if (_i % 1000 == 0)
			std::cout << "nms iter " << _i << std::endl;
		int i = order_ints[_i];
		if (not suppressed[i])
		{
			pick.push_back(i);
			for (int _j = _i + 1; _j < n; _j++) {
				int j = order_ints[_j];
				bool var_x = ((sides[i * 2 + 0] + sides[j * 2 + 0]) / 2 - abs(centers[i * 2 + 0] - centers[j * 2 + 0])) > 0;
				bool var_y = ((sides[i * 2 + 1] + sides[j * 2 + 1]) / 2 - abs(centers[i * 2 + 1] - centers[j * 2 + 1])) > 0;
				if (var_x and var_y and (not suppressed[i]))
				{
					c.AddPath(polygons[i], ptClip, true);
					c.AddPath(polygons[j], ptSubject, true);
					c.Execute(ctIntersection, solutions, pftNonZero, pftNonZero);
					double inter;
					if (solutions.size() > 0)
					{
						inter = Area(solutions[0]) / std::pow(4.0, 31);
						c.Clear();
					}
					else
						inter = 0.0;
					double union_ = (areas[i] + areas[j]) - inter;
					double iou = (union_ > 0.0) ? inter / union_ : 0.0;
					if (union_ > 0.0 and iou > overlapThresh)
						suppressed[j] = true;
					if (iou > neighbourThresh)
					{
						neighbours[neigh_cnt] = j;
						neigh_cnt += 1;
					}
				}
			}
			if (neigh_cnt >= num_neig)
			{
				neighbours[neigh_cnt] = i;
				neigh_cnt += 1;
				double temp_scores_sum = 0.0;
				for (int idx = 0; idx < neigh_cnt; idx++) // mul words NN scores by temp score 
				{
					double temp_score = boxes[neighbours[idx] * m + 8] - minScore;
					temp_scores_sum += temp_score;
					for (int jdx = 0; jdx < 8; jdx++)
						box_to_agg[idx * 8 + jdx] = boxes[neighbours[idx] * m + jdx] * temp_score;
				}
				for (int idx = 0; idx < 8; idx++) // mul coords by temp score 
				{
					double col_sum = 0.0;
					for (int jdx = 0; jdx < neigh_cnt; jdx++)
						col_sum += box_to_agg[jdx * 8 + idx];
					col_sum /= temp_scores_sum;
					new_boxes[i * m + idx] = col_sum;
				}
				new_boxes[i * m + 8] = boxes[i * m + 8];
			}
			else
			{
				for (int idx = 0; idx < neigh_cnt; idx++)
					suppressed[neighbours[idx]] = false;
				pick.pop_back();
			}
			neigh_cnt = 0;
		}
	}
	std::pair<vector<int>, float*> result = std::make_pair(pick, new_boxes);

	delete[] boxes;
	delete[] suppressed;
	delete[] areas;
	delete[] polygons;
	delete[] ref_scaled_polygon;
	delete[] centers;
	delete[] sides;
	delete[] order_floats;
	delete[] order_ints;
	delete[] box_to_agg;

	end = std::chrono::steady_clock::now();
	std::cout << "Time loop nms: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;


	return result;
}

void testNms(std::string boxes_f_name="nms_boxes.txt")
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	vector<std::string> vecOfStr;
	bool result = getFileContent(boxes_f_name, vecOfStr);
	vector<vector<float>> bboxes_test(vecOfStr.size(), vector<float>(9));
	if (result)
	{
		for (int i = 0; i < bboxes_test.size(); i++)
		{
			vector<std::string> splitted = split(vecOfStr[i], ',');
			for (int j = 0; j < splitted.size(); j++)
				bboxes_test[i][j] = ::atof(splitted[j].c_str());
		}
	}
	const int n = bboxes_test.size();
	const int m = bboxes_test[0].size();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time preparing data: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;

	begin = std::chrono::steady_clock::now();
	std::pair<vector<int>, float*> res = nms(bboxes_test, 0.15, 0.5, 0.0, 1);
	end = std::chrono::steady_clock::now();
	std::cout << "Time nms: " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 << std::endl;

	std::cout << "Printing 5 data pts" << std::endl;
	std::cout << "Picks" << std::endl;
	for (int i = 0; i < 5; i++)
		std::cout << res.first[i] << ' ';
	std::cout << std::endl;
	std::cout << "Boxes" << std::endl;
	for (int i = 0; i < 5; i++)
	{
		int idx = res.first[i];
		std::cout << "i: " << i;
		for (int j = 0; j < m; j++)
			std::cout << ' ' << std::setprecision(5) << res.second[idx * m + j] << ' ';
		std::cout << std::endl;
	}
	delete[] res.second;
}

int main(int argc, char* argv[])
{
	std::cout << "Starting nms" << std::endl;
	testNms();
	std::cout << std::endl;
}
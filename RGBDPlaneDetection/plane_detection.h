#ifndef PLANEDETECTION_H
#define PLANEDETECTION_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <Eigen/Eigen>
#include "include/peac/AHCPlaneFitter.hpp"
#include "include/MRF2.2/mrf.h"
#include "include/MRF2.2/GCoptimization.h"
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>


using namespace std;

typedef Eigen::Vector3d VertexType;
typedef Eigen::Vector2i PixelPos;

#if defined(__linux__) || defined(__APPLE__)
#define _isnan(x) isnan(x)
#endif

struct ImagePointCloud
{
	vector<VertexType> vertices; // 3D vertices
	int w, h;

	inline int width() const { return w; }
	inline int height() const { return h; }
	inline bool get(const int row, const int col, double &x, double &y, double &z) const {
		const int pixIdx = row * w + col;
		z = vertices[pixIdx][2];
		// Remove points with 0 or invalid depth in case they are detected as a plane
		if (z == 0 || _isnan(z)) return false;
		x = vertices[pixIdx][0];
		y = vertices[pixIdx][1];
		return true;
	}
};

// Data for sum of points on a same plane
struct SumStats
{
	double sx, sy, sz; // sum of x/y/z
	double sxx, syy, szz, sxy, syz, sxz; // sum of x*x/y*y/z*z/x*y/y*z/x*z
	SumStats(){
		sx = sy = sz = sxx = syy = szz = sxy = syz = sxz = 0;
	}
};

namespace py = pybind11;

class PlaneDetection
{
public:
	ImagePointCloud cloud;
	ahc::PlaneFitter< ImagePointCloud > plane_filter;
	vector<vector<int>> plane_vertices_; // vertex indices each plane contains
	cv::Mat seg_img_; // segmentation image
	cv::Mat color_img_; // input color image
	int plane_num_;
	
	
	// Camera intrinsic parameters.
	// All BundleFusion data uses the following parameters.
	double fx_;
	double fy_;
	double cx_;
	double cy_;
	int depthWidth_;
	int depthHeight_;
	int depthImageZScaleFactor_; // scale coordinate unit in mm

	/* For MRF optimization */
	int neighborRange_; // boundary pixels' neighbor range in MRF
	float infVal_; // an infinite large value used in MRF optimization

	cv::Mat opt_seg_img_;
	cv::Mat opt_membership_img_; // optimized membership image (plane index each pixel belongs to)
	vector<bool> pixel_boundary_flags_; // pixel is a plane boundary pixel or not
	vector<int> pixel_grayval_;
	vector<cv::Vec3b> plane_colors_;
	vector<SumStats> sum_stats_, opt_sum_stats_; // parameters of sum of points from the same plane
	vector<int> plane_pixel_nums_, opt_plane_pixel_nums_; // number of pixels each plane has
	unordered_map<int, int> pid_to_extractedpid; // plane index -> extracted plane index of plane_filter.extractedPlanes
	unordered_map<int, int> extractedpid_to_pid; // extracted plane index -> plane index

public:
	PlaneDetection();
	~PlaneDetection();

	//bool readIntrinsicParameterFile(string filename);

	// bool readColorImage(string filename);
	//
	// bool readDepthImage(string filename);

	bool setColorImage(const py::array_t<uchar>& img);
	bool setDepthImage(const py::array_t<uint16_t>& img);
	// plane fitter params 
	void setPlaneFitterMinSupport(int minSupport); // minimum number of points on a plane Todo: not clear what units this is 
	void setPlaneFitterMaxStep(int maxStep);
	void setPlaneFitterWindowSize(int windowWidth, int windowHeight);
	// Camera intrinsic parameters.
	void setCameraIntrinsic(double fx, double fy, double cx, double cy);
	void setDepthImageParams(int width, int height, double imageZScaleFactor);
	// MRF optimization params
	void setMrfParams(int neighborRange, float infVal);

	py::array_t<uint8_t>  getMembershipImg();
	py::array_t<uint8_t>  getSegImgOptimized();
	py::array_t<uint8_t>  getSegImg();
	py::array_t<double>  getPlaneNormals();
	py::array_t<double> getPlaneCenters();

	bool runPlaneDetection();

	void prepareForMRF();

	cv::Mat deepCopy(const cv::Mat& orig_mat);

	cv::Mat pyarray_to_cvmat_color(const py::array_t<uint8_t>& input);
	cv::Mat pyarray_to_cvmat_gray(const py::array_t<uint16_t>& input);
	void runMRFOptimization();

	py::array_t<uint8_t> cvmat_to_pyarray(cv::Mat& image);

	

	// void writeOutputFiles(string output_folder, string frame_name, bool run_mrf = false);
	//
	// void writePlaneDataFile(string filename, bool run_mrf = false);
	//
	// void writePlaneLabelFile(string filename, bool run_mrf = false);

	// /************************************************************************/
	// /* For MRF optimization */
	// inline MRF::CostVal dCost(int pix, int label)
	// {
	// 	return pixel_boundary_flags_[pix] ? 1 : (label == plane_filter.membershipImg.at<int>(pix / depthWidth_, pix % depthWidth_) ? 1 : kInfVal);
	// }
	// inline MRF::CostVal fnCost(int pix1, int pix2, int i, int j)
	// {
	// 	int gray1 = pixel_grayval_[pix1], gray2 = pixel_grayval_[pix2];
	// 	return i == j ? 0 : exp(-MRF::CostVal(gray1 - gray2) * (gray1 - gray2) / 900); // 900 = sigma^2 by default
	// }
	// /************************************************************************/

	MRF::CostVal dCost(int pix, int label);
	MRF::CostVal fnCost(int pix1, int pix2, int i, int j);

private:
	inline int RGB2Gray(int x, int y)
	{
		return int(0.299 * color_img_.at<cv::Vec3b>(x, y)[2] +
			0.587 * color_img_.at<cv::Vec3b>(x, y)[1] +
			0.114 * color_img_.at<cv::Vec3b>(x, y)[0] +
			0.5);
	}

	void computePlaneSumStats(bool run_mrf = false);

};

#endif
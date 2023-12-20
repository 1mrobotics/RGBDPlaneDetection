#include "plane_detection.h"
#include "opencv2/opencv.hpp"
#include <stdint.h>
#include <iomanip> // output double value precision

namespace py = pybind11;

// -------------------- Python bindings: ----------------------------


PlaneDetection::PlaneDetection()
{
	cloud.vertices.resize(kDepthHeight * kDepthWidth);
	cloud.w = kDepthWidth;
	cloud.h = kDepthHeight;
}

PlaneDetection::~PlaneDetection()
{
	cloud.vertices.clear();
	seg_img_.release();
	opt_seg_img_.release();
	color_img_.release();
	opt_membership_img_.release();
	pixel_boundary_flags_.clear();
	pixel_grayval_.clear();
	plane_colors_.clear();
	plane_pixel_nums_.clear();
	opt_plane_pixel_nums_.clear();
	sum_stats_.clear();
	opt_sum_stats_.clear();

}

cv::Mat PlaneDetection::pyarray_to_cvmat_color(const py::array_t<uint8_t>& input) {
    // Get information about the NumPy array
    py::buffer_info buf_info = input.request();

    // Check if the array has three dimensions
    if (buf_info.ndim != 3 || buf_info.shape[2] != 3) {
        throw std::runtime_error("Input array must have three dimensions (height x width x channels).");
    }

	// Ensure proper alignment by calculating the aligned step
    size_t esz1 = buf_info.itemsize;  // size of one element in bytes
    size_t step = buf_info.strides[0];  // original step

    if (step % esz1 != 0) {
        size_t aligned_step = (step / esz1 + 1) * esz1;
        step = aligned_step;
    }

    // Create a cv::Mat using the data pointer from the NumPy array
    cv::Mat cv_mat = cv::Mat(cv::Size(buf_info.shape[1], buf_info.shape[0]), CV_8UC3, buf_info.ptr, step);
	cout << "cv mat size is " << cv_mat.size() << endl;
	cv::imwrite("cpp_color.png", cv_mat);
    return cv_mat; // cv_mat.clone();  // Clone the matrix to ensure data ownershipz
}

cv::Mat PlaneDetection::pyarray_to_cvmat_gray(const py::array_t<uint16_t>& input) {
    // Get information about the NumPy array
    py::buffer_info buf_info = input.request();

    // Check if the array has two dimensions
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Input array must have two dimensions (height x width).");
    }
	// Extract the data pointer and shape information
	uint16_t *ptr = static_cast<uint16_t *>(buf_info.ptr);
	int height = buf_info.shape[0];
	int width = buf_info.shape[1];

 //
	// // Ensure proper alignment by calculating the aligned step
 size_t esz1 = buf_info.itemsize;  // size of one element in bytes
 size_t step = buf_info.strides[0];  // original step

 if (step % esz1 != 0) {
     size_t aligned_step = (step / esz1 + 1) * esz1;
     step = aligned_step;
 }


    // Create a cv::Mat using the data pointer from the NumPy array
    cv::Mat cv_mat(height, width, CV_16UC1, ptr, step);
    // cv::Mat cv_mat = cv::Mat(cv::Size(buf_info.shape[1], buf_info.shape[0]), CV_16UC1, buf_info.ptr, step);
	cv::imwrite("cpp_depth.png", cv_mat);
    return cv_mat; // cv_mat.clone();  // Clone the matrix to ensure data ownership
}


py::array_t<uint8_t> PlaneDetection::cvmat_to_pyarray(cv::Mat& image) {
    int width = image.cols;
    int height = image.rows;

    // Create a NumPy array
    py::array_t<uint8_t> array({ height, width, image.channels() }, image.data);

    return array;
}


bool PlaneDetection::setColorImage(const py::array_t<uchar>& img)
{
	color_img_ =  pyarray_to_cvmat_color(img);
	if (color_img_.empty()){
		cout << "ERROR: cannot read color image. No such a file" << endl;
		return false;
	}
	if (color_img_.depth() != CV_8U){
		cout << "ERROR: cannot read color image. The image format is not 8UC3" << endl;
		return false;
	}
	return true;
}

/************************************************************************/
/* For MRF optimization */
MRF::CostVal PlaneDetection::dCost(int pix, int label)
{
	return pixel_boundary_flags_[pix] ? 1 : (label == plane_filter.membershipImg.at<int>(pix / kDepthWidth, pix % kDepthWidth) ? 1 : kInfVal);
}
MRF::CostVal PlaneDetection::fnCost(int pix1, int pix2, int i, int j)
{
	int gray1 = pixel_grayval_[pix1], gray2 = pixel_grayval_[pix2];
	return i == j ? 0 : exp(-MRF::CostVal(gray1 - gray2) * (gray1 - gray2) / 900); // 900 = sigma^2 by default
}
/************************************************************************/


bool PlaneDetection::setDepthImage(const py::array_t<uint16_t>& img)
{
	cv::Mat depth_img = pyarray_to_cvmat_gray(img);
	// if (depth_img.empty() || depth_img.depth() != CV_16U)
	if (depth_img.empty()){
		cout << "ERROR: cannot read color image. No such a file" << endl;
		return false;
	}
	if (depth_img.depth() != CV_16U)
	{
		cout << "WARNING: cannot read depth image. The image format is not 16UC1" << endl;
		return false;
	}
	int rows = depth_img.rows, cols = depth_img.cols;
	int vertex_idx = 0;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			double z = (double)(depth_img.at<unsigned short>(i, j)) / kScaleFactor;
			if (_isnan(z))
			{
				cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
			double x = ((double)j - kCx) * z / kFx;
			double y = ((double)i - kCy) * z / kFy;
			cloud.vertices[vertex_idx++] = VertexType(x, y, z);
		}
	}
	return true;
}

py::array_t<uint8_t> PlaneDetection::getMembershipImg() {
	return cvmat_to_pyarray(opt_membership_img_); // optimized membership image (plane index each pixel belongs to)
}

py::array_t<uint8_t> PlaneDetection::getSegImg() {
	return cvmat_to_pyarray(opt_seg_img_); // optimized membership image (plane index each pixel belongs to)
}

py::array_t<double> PlaneDetection::getPlaneNormals()
{	
	// Create a vector of double arrays
	std::vector<double> doubleVector;

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		doubleVector.insert(doubleVector.end(), std::begin(plane_filter.extractedPlanes[pidx]->normal), std::end(plane_filter.extractedPlanes[pidx]->normal));
	}
	py::array_t<double> normals({plane_num_, 3}, &doubleVector[0]);
	return normals;
}

py::array_t<double> PlaneDetection::getPlaneCenters()
{	
	// Create a vector of double arrays
	std::vector<double> doubleVector;

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		doubleVector.insert(doubleVector.end(), std::begin(plane_filter.extractedPlanes[pidx]->center), std::end(plane_filter.extractedPlanes[pidx]->center));
	}
	py::array_t<double> centers({plane_num_, 3}, &doubleVector[0]);
	return centers;
}

void PlaneDetection::setMinSupport(int minSupport)
{
	min_support_ = minSupport;
}

void PlaneDetection::setMaxStep(int maxStep)
{
	//max number of steps for merging clusters
	max_step_ = maxStep;
}

void PlaneDetection::setWindowSize(int width, int height)
{
	//make sure width is divisible by windowWidth
	//similarly for height and windowHeight
	window_height_ = height;
	window_width_ = width; 
}

bool PlaneDetection::runPlaneDetection()
{
	plane_filter.minSupport = min_support_; 
	seg_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_8UC3);
	plane_filter.run(&cloud, &plane_vertices_, &seg_img_);
	plane_num_ = (int)plane_vertices_.size();

	// Here we set the plane index of a pixel which does NOT belong to any plane as #planes.
	// This is for using MRF optimization later.
	for (int row = 0; row < kDepthHeight; ++row)
		for (int col = 0; col < kDepthWidth; ++col)
			if (plane_filter.membershipImg.at<int>(row, col) < 0)
				plane_filter.membershipImg.at<int>(row, col) = plane_num_;
	return true;
}

void PlaneDetection::prepareForMRF()
{
	opt_seg_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_8UC3);
	opt_membership_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_32SC1);
	pixel_boundary_flags_.resize(kDepthWidth * kDepthHeight, false);
	pixel_grayval_.resize(kDepthWidth * kDepthHeight, 0);
	cv::Mat& mat_label = plane_filter.membershipImg;

	for (int row = 0; row < kDepthHeight; ++row)
	{
		for (int col = 0; col < kDepthWidth; ++col)
		{
			pixel_grayval_[row * kDepthWidth + col] = RGB2Gray(row, col);
			int label = mat_label.at<int>(row, col);
			// todo This is throwing an error
			if ((row - 1 >= 0 && mat_label.at<int>(row - 1, col) != label)
				|| (row + 1 < kDepthHeight && mat_label.at<int>(row + 1, col) != label)
				|| (col - 1 >= 0 && mat_label.at<int>(row, col - 1) != label)
				|| (col + 1 < kDepthWidth && mat_label.at<int>(row, col + 1) != label))
			{
				// Pixels in a fixed range near the boundary pixel are also regarded as boundary pixels
				for (int x = max(row - kNeighborRange, 0); x < min(kDepthHeight, row + kNeighborRange); ++x)
				{
					for (int y = max(col - kNeighborRange, 0); y < min(kDepthWidth, col + kNeighborRange); ++y)
					{
						// If a pixel is not on any plane, then it is not a boundary pixel.
						if (mat_label.at<int>(x, y) == plane_num_)
							continue;
						pixel_boundary_flags_[x * kDepthWidth + y] = true;
					}
				}
			}
		}
	}

	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		int vidx = plane_vertices_[pidx][0];
		cv::Vec3b c = seg_img_.at<cv::Vec3b>(vidx / kDepthWidth, vidx % kDepthWidth);
		plane_colors_.push_back(c);
	}

	plane_colors_.push_back(cv::Vec3b(0,0,0)); // black for pixels not in any plane
}


void PlaneDetection::computePlaneSumStats(bool run_mrf /* = false */)
{
	sum_stats_.resize(plane_num_);
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		for (int i = 0; i < plane_vertices_[pidx].size(); ++i)
		{
			int vidx = plane_vertices_[pidx][i];
			const VertexType& v = cloud.vertices[vidx];
			sum_stats_[pidx].sx += v[0];		 sum_stats_[pidx].sy += v[1];		  sum_stats_[pidx].sz += v[2];
			sum_stats_[pidx].sxx += v[0] * v[0]; sum_stats_[pidx].syy += v[1] * v[1]; sum_stats_[pidx].szz += v[2] * v[2];
			sum_stats_[pidx].sxy += v[0] * v[1]; sum_stats_[pidx].syz += v[1] * v[2]; sum_stats_[pidx].sxz += v[0] * v[2];
		}
		plane_pixel_nums_.push_back(int(plane_vertices_[pidx].size()));
	}
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		int num = plane_pixel_nums_[pidx];
		sum_stats_[pidx].sx /= num;		sum_stats_[pidx].sy /= num;		sum_stats_[pidx].sz /= num;
		sum_stats_[pidx].sxx /= num;	sum_stats_[pidx].syy /= num;	sum_stats_[pidx].szz /= num;
		sum_stats_[pidx].sxy /= num;	sum_stats_[pidx].syz /= num;	sum_stats_[pidx].sxz /= num;
	}
	// Note that the order of extracted planes in `plane_filter.extractedPlanes` is DIFFERENT from
	// the plane order in `plane_vertices_` after running plane detection function `plane_filter.run()`.
	// So here we compute a mapping between these two types of plane indices by comparing plane centers.
	vector<double> sx(plane_num_), sy(plane_num_), sz(plane_num_);
	for (int i = 0; i < plane_filter.extractedPlanes.size(); ++i)
	{
		sx[i] = plane_filter.extractedPlanes[i]->stats.sx / plane_filter.extractedPlanes[i]->stats.N;
		sy[i] = plane_filter.extractedPlanes[i]->stats.sy / plane_filter.extractedPlanes[i]->stats.N;
		sz[i] = plane_filter.extractedPlanes[i]->stats.sz / plane_filter.extractedPlanes[i]->stats.N;
	}
	extractedpid_to_pid.clear();
	pid_to_extractedpid.clear();
	// If two planes' centers are closest, then the two planes are corresponding to each other.
	for (int i = 0; i < plane_num_; ++i)
	{
		double min_dis = 1000000;
		int min_idx = -1;
		for (int j = 0; j < plane_num_; ++j)
		{
			double a = sum_stats_[i].sx - sx[j], b = sum_stats_[i].sy - sy[j], c = sum_stats_[i].sz - sz[j];
			double dis = a * a + b * b + c * c;
			if (dis < min_dis)
			{
				min_dis = dis;
				min_idx = j;
			}
		}
		if (extractedpid_to_pid.find(min_idx) != extractedpid_to_pid.end())
		{
			cout << "   WARNING: a mapping already exists for extracted plane " << min_idx << ":" << extractedpid_to_pid[min_idx] << " -> " << min_idx << endl;
		}
		pid_to_extractedpid[i] = min_idx;
		extractedpid_to_pid[min_idx] = i;
	}
	if (run_mrf)
	{
		opt_sum_stats_.resize(plane_num_);
		opt_plane_pixel_nums_.resize(plane_num_, 0);
		for (int row = 0; row < kDepthHeight; ++row)
		{
			for (int col = 0; col < kDepthWidth; ++col)
			{
				int label = opt_membership_img_.at<int>(row, col); // plane label each pixel belongs to
				if (label != plane_num_) // pixel belongs to some plane
				{
					opt_plane_pixel_nums_[label]++;
					int vidx = row * kDepthWidth + col;
					const VertexType& v = cloud.vertices[vidx];
					opt_sum_stats_[label].sx += v[0];		  opt_sum_stats_[label].sy += v[1];		    opt_sum_stats_[label].sz += v[2];
					opt_sum_stats_[label].sxx += v[0] * v[0]; opt_sum_stats_[label].syy += v[1] * v[1]; opt_sum_stats_[label].szz += v[2] * v[2];
					opt_sum_stats_[label].sxy += v[0] * v[1]; opt_sum_stats_[label].syz += v[1] * v[2]; opt_sum_stats_[label].sxz += v[0] * v[2];
				}
			}
		}
		for (int pidx = 0; pidx < plane_num_; ++pidx)
		{
			int num = opt_plane_pixel_nums_[pidx];
			opt_sum_stats_[pidx].sx /= num;		opt_sum_stats_[pidx].sy /= num;		opt_sum_stats_[pidx].sz /= num;
			opt_sum_stats_[pidx].sxx /= num;	opt_sum_stats_[pidx].syy /= num;	opt_sum_stats_[pidx].szz /= num;
			opt_sum_stats_[pidx].sxy /= num;	opt_sum_stats_[pidx].syz /= num;	opt_sum_stats_[pidx].sxz /= num;
		}
	}

	//--------------------------------------------------------------
	// Only for debug. It doesn't influence the plane detection.
	for (int pidx = 0; pidx < plane_num_; ++pidx)
	{
		double w = 0;
		//for (int j = 0; j < 3; ++j)
		//	w -= plane_filter.extractedPlanes[pidx]->normal[j] * plane_filter.extractedPlanes[pidx]->center[j];
		w -= plane_filter.extractedPlanes[pidx]->normal[0] * sum_stats_[pidx].sx;
		w -= plane_filter.extractedPlanes[pidx]->normal[1] * sum_stats_[pidx].sy;
		w -= plane_filter.extractedPlanes[pidx]->normal[2] * sum_stats_[pidx].sz;
		double sum = 0;
		for (int i = 0; i < plane_vertices_[pidx].size(); ++i)
		{
			int vidx = plane_vertices_[pidx][i];
			const VertexType& v = cloud.vertices[vidx];
			double dis = w;
			for (int j = 0; j < 3; ++j)
				dis += v[j] * plane_filter.extractedPlanes[pidx]->normal[j];
			sum += dis * dis;
		}
		sum /= plane_vertices_[pidx].size();
		cout << "Distance for plane " << pidx << ": " << sum << endl;
	}
}


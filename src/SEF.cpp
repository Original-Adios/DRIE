#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iomanip>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

#include "image_enhancement.h"
using namespace std;

double getImagePixelMaxValue(cv::Mat img)
{
	double minVal, maxVal;
	cv::minMaxLoc(img, &minVal, &maxVal);
	return maxVal;
}

void saveGeneratedImgSeq(cv::Mat img, int i, string save_path)
{
	string image_name = "imgSeq" + to_string(i) + ".tif";
	string save_location = save_path + "/" + image_name;
	cout << "转换成16位单通道图像之前" << image_name << "的最大像素值为：" << getImagePixelMaxValue(img) << endl;
	img.convertTo(img, CV_16U, 65535); // 将生成的图像序列转换成16位图像进行保存
	cout << "转换成16位单通道图像之后生成的图像序列" << i << "的图像信息为：" << endl;
	printImgInformation(img);
	cv::imwrite(save_location, img); // 保存生成的图像序列
	cout << image_name << "保存成功！！！" << endl;
}

void printImgInformation(cv::Mat img)
{
	int type = img.type();
	int channels = img.channels();
	string typestr;
	switch (type)
	{
	case CV_8UC1:
		typestr = "8位单通道";
		break;
	case CV_8UC3:
		typestr = "8位三通道";
		break;
	case CV_16UC1:
		typestr = "16位单通道";
		break;
	case CV_16UC3:
		typestr = "16位三通道";
		break;
	case CV_64FC1:
		typestr = "64FC1";
		break;
	case CV_64FC3:
		typestr = "64FC3";
		break;
	default:
		typestr = "unknown";
		break;
	}
	cout << "图像的像素类型为：" << typestr << "，通道数为：" << channels << endl;
	cout << "最大像素值为：" << getImagePixelMaxValue(img) << endl;
	cout << "-------------------------------------------" << endl;

	return;
}

string ToString(double val) // doule转string
{
	stringstream ss;
	ss << setiosflags(ios::fixed) << setprecision(2) << val; // 保留两位小数
	string str = ss.str();
	return str;
}

std::vector<cv::Mat> gaussian_pyramid(const cv::Mat &src, int nLevel) // 高斯金字塔
{
	cv::Mat I = src.clone();
	std::vector<cv::Mat> pyr;
	pyr.push_back(I); // push_back() 在Vector最后添加一个元素（参数为要插入的值）
	for (int i = 2; i <= nLevel; i++)
	{
		cv::pyrDown(I, I);
		pyr.push_back(I);
	}
	return pyr;
}

std::vector<cv::Mat> laplacian_pyramid(const cv::Mat &src, int nLevel) // 生成拉普拉斯金字塔
{
	cv::Mat I = src.clone();  // 将输入图像src进行复制，生成一个新的cv::Mat对象I。这样做是为了避免在函数中修改原始输入图像。
	std::vector<cv::Mat> pyr; // 定义了一个空的std::vector<cv::Mat>对象pyr
	cv::Mat J = I.clone();	  // 定义了一个新的cv::Mat对象J，它是I的复制品。J将在后面的循环中用作上一级别的图像。
	for (int i = 1; i < nLevel; i++)
	{
		cv::pyrDown(J, I); // 降采样 (cv::pyrDown)  J--输入图像（Mat类的对象即可）  I--输出图像
		cv::Mat J_up;
		cv::pyrUp(I, J_up, J.size()); // 上采样(cv::pyrUp)  I--输入图像  J_up--输出图像
		pyr.push_back(J - J_up);
		J = I;
	}
	pyr.push_back(J); // the coarest level contains the residual low pass image
	return pyr; // 回包含所有拉普拉斯金字塔图像的向量pyr
}
cv::Mat reconstruct_laplacian_pyramid(const std::vector<cv::Mat> &pyr)
{
	int nLevel = pyr.size();
	cv::Mat R = pyr[nLevel - 1].clone();
	for (int i = nLevel - 2; i >= 0; i--)
	{
		cv::pyrUp(R, R, pyr[i].size());
		R = pyr[i] + R;
	}
	return R;
}

cv::Mat multiscale_blending(const std::vector<cv::Mat> &seq, const std::vector<cv::Mat> &W)
{
	int h = seq[0].rows;
	int w = seq[0].cols;
	int n = seq.size();
	int nScRef = int(std::log(std::min(h, w)) / log(2));

	int nScales = 1; // 金字塔层数nScales
	int hp = h;		 // 每个层次的高度hp
	int wp = w;		 // 每个层次的宽度wp
	while (nScales < nScRef)
	{
		nScales++;
		hp = (hp + 1) / 2;
		wp = (wp + 1) / 2;
	}
	std::vector<cv::Mat> pyr;
	hp = h;
	wp = w;
	for (int scale = 1; scale <= nScales; scale++)
	{
		pyr.push_back(cv::Mat::zeros(hp, wp, CV_64F));
		hp = (hp + 1) / 2;
		wp = (wp + 1) / 2;
	}
	for (int i = 0; i < n; i++)
	{
		std::vector<cv::Mat> pyrW = gaussian_pyramid(W[i], nScales);
		std::vector<cv::Mat> pyrI = laplacian_pyramid(seq[i], nScales);

		for (int scale = 0; scale < nScales; scale++)
		{
			pyr[scale] += pyrW[scale].mul(pyrI[scale]);
		}
	}

	return reconstruct_laplacian_pyramid(pyr);
}
// 鲁棒归一化
void robust_normalization(const cv::Mat &src, cv::Mat &dst, double wSat = 1.0, double bSat = 1.0)
{
	int H = src.rows;		// 输入图像的高
	int W = src.cols;		// 输入图像的宽
	int D = src.channels(); // 输入图像的通道数
	int N = H * W;			// 像素总数
	double vmax;
	double vmin;
	if (D > 1)
	{
		std::vector<cv::Mat> src_channels;
		cv::split(src, src_channels);

		cv::Mat max_channel;
		cv::max(src_channels[0], src_channels[1], max_channel);
		cv::max(max_channel, src_channels[2], max_channel);
		cv::Mat max_channel_sort;
		cv::sort(max_channel.reshape(1, 1), max_channel_sort, cv::SORT_ASCENDING);
		vmax = max_channel_sort.at<double>(int(N - wSat * N / 100 + 1));

		cv::Mat min_channel;
		cv::min(src_channels[0], src_channels[1], min_channel);
		cv::min(min_channel, src_channels[2], min_channel);
		cv::Mat min_channel_sort;
		cv::sort(min_channel.reshape(1, 1), min_channel_sort, cv::SORT_ASCENDING);
		vmin = min_channel_sort.at<double>(int(bSat * N / 100));
	}
	else
	{
		cv::Mat src_sort;
		cv::sort(src.reshape(1, 1), src_sort, cv::SORT_ASCENDING);
		vmax = src_sort.at<double>(int(N - wSat * N / 100 + 1));
		vmin = src_sort.at<double>(int(bSat * N / 100));

		/*
		得到图像像素的最大最小值
		*/
	}

	if (vmax <= vmin)
	{
		if (D > 1)
			dst = cv::Mat(H, W, src.type(), cv::Scalar(vmax, vmax, vmax));
		else
			dst = cv::Mat(H, W, src.type(), cv::Scalar(vmax));
	}
	else
	{
		cv::Scalar Ones;
		if (D > 1)
		{
			cv::Mat vmin3 = cv::Mat(H, W, src.type(), cv::Scalar(vmin, vmin, vmin));
			cv::Mat vmax3 = cv::Mat(H, W, src.type(), cv::Scalar(vmax, vmax, vmax));
			dst = (src - vmin3).mul(1.0 / (vmax3 - vmin3));
			Ones = cv::Scalar(1.0, 1.0, 1.0);
		}
		else
		{
			dst = (src - vmin) / (vmax - vmin); // 图像归一化
			Ones = cv::Scalar(1.0);
		}

		cv::Mat mask_over = dst > vmax;
		cv::Mat mask_below = dst < vmin;
		mask_over.convertTo(mask_over, CV_64F, 1.0 / 255.0);
		mask_below.convertTo(mask_below, CV_64F, 1.0 / 255.0);

		dst = dst.mul(Ones - mask_over) + mask_over;
		dst = dst.mul(Ones - mask_below);
	}

	return;
}

void saveWeightImgSeqAsHeatMap(cv::Mat img, int i, string save_path)
{
	string image_name = "WeightImgSeq" + to_string(i) + ".png";
	string save_location = save_path + "/" + image_name;
	// 将权重图线性映射到0-255范围
	cv::Mat scaledHeatmap;
	img.convertTo(scaledHeatmap, CV_8UC1, 255.0);
	// 将灰度图转换为伪彩色图像
	cv::Mat colorMap;
	cv::applyColorMap(scaledHeatmap, colorMap, cv::COLORMAP_JET);
	printImgInformation(colorMap);
	cv::imwrite(save_location, colorMap); // 保存生成的权重图像序列
	cout << image_name << "保存成功！！！" << endl;
	cout << "-------------------------------------------" << endl;
}

void SEF(const cv::Mat &src, cv::Mat &dst, double alpha, double beta, double lambda)
{
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	string dir_name = "α=" + ToString(alpha) + "，" + "β=" + ToString(beta);
	string dir_path = "../" + dir_name;
	if (!fs::exists(dir_path))
	{
		fs::create_directory(dir_path);
	}
	string allImageFile = "result";
	string all_save_path = "../" + allImageFile;
	if (!fs::exists(all_save_path))
	{
		fs::create_directory(all_save_path);
	}
	std::cout << "α=" << alpha << "    "
			  << "β=" << beta << std::endl;
	int rows = src.rows;
	int cols = src.cols;
	cout << "原图像行数：" << rows
		 << "列数：" << cols
		 << "总像素数：" << rows * cols
		 << endl;
	int channels = src.channels();
	printImgInformation(src);

	cv::Mat L;
	cv::Mat HSV;
	std::vector<cv::Mat> HSV_channels;
	if (channels == 1)
	{
		L = src.clone(); // 在这把原图像复制了一张
	}
	else
	{
		cv::cvtColor(src, HSV, cv::COLOR_BGR2HSV_FULL);
		cv::split(HSV, HSV_channels);
		L = HSV_channels[2];
	}
	cv::Mat L_norm; // 把原图像的所有像素压缩到0~1
	L.convertTo(L_norm, CV_64F, 1.0 / 65535.0);
	cv::Mat src_norm;
	src.convertTo(src_norm, CV_64F, 1.0 / 65535.0);
	cout << "原图像归一化后的信息：" << endl;
	cv::Mat C;
	if (channels == 1)
	{
		C = src_norm.mul(1.0 / (L_norm + std::pow(2, -16)));
		cout << "*****************************************" << endl;
		cout << "将原图像的像素值归一化到[0, 1]之间后对每个像素值除以该像素值加上一个很小的值。" << endl;
		cout << "原理：加上一个很小的常数可以使得图像中的小值更容易被拉升，增强图像的对比度" << endl;
		cout << "原图像归一化后增强对比度的最大像素值为: " << getImagePixelMaxValue(C) << endl;
		cout << "*****************************************" << endl;
	}
	else
	{
		cv::Mat temp = 1.0 / (L_norm + std::pow(2, -16));
		std::vector<cv::Mat> temp_arr = {temp.clone(), temp.clone(), temp.clone()};
		cv::Mat temp3;
		cv::merge(temp_arr, temp3);
		C = src_norm.mul(temp3);
	}
	// 计算中值
	cv::Mat tmp = src.reshape(1, 1); // 将输入图像src转换为一个大小为1x(total_pixels*channels)的矩阵，其中每一行代表输入图像中的一个像素，每个通道都在同一行中排列。
	cv::Mat sorted;
	cv::sort(tmp, sorted, cv::SORT_ASCENDING);
	double med = double(sorted.at<ushort>(0, rows * cols * channels / 2)) / 65535.0; // 计算排序后的矩阵的中位数，并将其存储在med中。由于输入图像已经被归一化到0到1之间，因此计算中位数时需要将结果除以255.0
	double maxValue = double(sorted.at<ushort>(0, rows * cols * channels - 1)) / 65535.0;
	cout << "############################################" << endl;
	std::cout << "原图像像素中值= " << med << std::endl;
	std::cout << "原图像像素最大值= " << maxValue << std::endl;

	// 计算最优的图像数量
	int Mp = 1;																					  // Mp代表总共需要生成的图像数量
	int Ns = int(Mp * med);																		  // Ns代表使用fs函数生成的图像数量
	int N = Mp - Ns;																			  // N代表使用f函数生成的图像数量
	int Nx = std::max(N, Ns);																	  // Nx用于计算最大因子
	double tmax1 = (1.0 + (Ns + 1.0) * (beta - 1.0) / Mp) / (std::pow(alpha, 1.0 / Nx));		  // t_max k=+1
	double tmin1s = (-beta + (Ns - 1.0) * (beta - 1.0) / Mp) / (std::pow(alpha, 1.0 / Nx)) + 1.0; // t_min k=-1
	double tmax0 = 1.0 + Ns * (beta - 1.0) / Mp;												  // t_max k=0
	double tmin0 = 1.0 - beta + Ns * (beta - 1.0) / Mp;											  // t_min k=0
	while (tmax1 < tmin0 || tmax0 < tmin1s)
	{
		Mp++;
		Ns = int(Mp * med);
		N = Mp - Ns;
		Nx = std::max(N, Ns);
		tmax1 = (1.0 + (Ns + 1.0) * (beta - 1.0) / Mp) / (std::pow(alpha, 1.0 / Nx));
		tmin1s = (-beta + (Ns - 1.0) * (beta - 1.0) / Mp) / (std::pow(alpha, 1.0 / Nx)) + 1.0;
		tmax0 = 1.0 + Ns * (beta - 1.0) / Mp;
		tmin0 = 1.0 - beta + Ns * (beta - 1.0) / Mp;
		if (Mp > 49)
		{
			std::cerr << "The estimation of the number of image required in the sequence stopped, please check the parameters!" << std::endl;
		}
	}

	// std::cout << "M = " << Mp + 1 << ", with N = " << N << " and Ns = " << Ns << std::endl;
	std::cout << "需要融合的总图像数量M = " << Mp + 1 << "，曝光过度图像数量N = " << N << "，曝光不足图像数量Ns = " << Ns << std::endl;
	cout << "############################################" << endl;
	// Remapping functions
	auto fun_f = [alpha, Nx](cv::Mat t, int k) {  // enhance dark parts增强暗部
		return std::pow(alpha, k * 1.0 / Nx) * t; // 公式9
	};
	auto fun_fs = [alpha, Nx](cv::Mat t, int k) {				 // enhance bright parts增强亮部
		return std::pow(alpha, -k * 1.0 / Nx) * (t - 1.0) + 1.0; // 公式9
	};

	// Offset for the dynamic range reduction (function "g")
	auto fun_r = [beta, Ns, Mp](int k) // 公式7
	{
		return (1.0 - beta / 2.0) - (k + Ns) * (1.0 - beta) / Mp;
	};
	double a = beta / 2 + lambda; // 公式8下方的解释
	double b = beta / 2 - lambda;
	auto fun_g = [fun_r, beta, a, b, lambda](cv::Mat t, int k)
	{
		auto rk = fun_r(k);
		cv::Mat diff = t - rk;
		cv::Mat abs_diff = cv::abs(diff);

		cv::Mat mask = abs_diff <= beta / 2; // mask用来标记亮度没有超过范围的像素位置
		mask.convertTo(mask, CV_64F, 1.0 / 255.0);
		cv::Mat sign = diff.mul(1.0 / abs_diff);

		return mask.mul(t) + (1.0 - mask).mul(sign.mul(a - lambda * lambda / (abs_diff - b)) + rk); // 公式8
	};
	auto fun_h = [fun_f, fun_g](cv::Mat t, int k) { // create brighter images (k>=0) (enhance dark parts)
		return fun_g(fun_f(t, k), k);
	};
	auto fun_hs = [fun_fs, fun_g](cv::Mat t, int k) { // create darker images (k<0) (enhance bright parts)
		return fun_g(fun_fs(t, k), k);
	};
	auto fun_dg = [fun_r, beta, b, lambda](cv::Mat t, int k)
	{
		auto rk = fun_r(k);
		cv::Mat diff = t - rk;
		cv::Mat abs_diff = cv::abs(diff);

		cv::Mat mask = abs_diff <= beta / 2;
		mask.convertTo(mask, CV_64F, 1.0 / 255.0);

		cv::Mat p;
		cv::pow(abs_diff - b, 2, p);

		return mask + (1.0 - mask).mul(lambda * lambda / p);
	};
	auto fun_dh = [alpha, Nx, fun_f, fun_dg](cv::Mat t, int k)
	{
		return std::pow(alpha, k * 1.0 / Nx) * fun_dg(fun_f(t, k), k);
	};
	auto fun_dhs = [alpha, Nx, fun_fs, fun_dg](cv::Mat t, int k)
	{
		return std::pow(alpha, -k * 1.0 / Nx) * fun_dg(fun_fs(t, k), k);
	};

	// Simulate a sequence from image L_norm and compute the contrast weights
	std::vector<cv::Mat> seq(N + Ns + 1);
	std::vector<cv::Mat> wc(N + Ns + 1);
	for (int k = -Ns; k <= N; k++)
	{
		cv::Mat seq_temp, wc_temp;
		if (k < 0)
		{
			seq_temp = fun_hs(L_norm, k); // Apply remapping function
			wc_temp = fun_dhs(L_norm, k); // Compute contrast measure
		}
		else
		{
			seq_temp = fun_h(L_norm, k); // Apply remapping function
			wc_temp = fun_dh(L_norm, k); // Compute contrast measure
		}

		cout << "权重图的图像类型为：" << endl;
		printImgInformation(wc_temp);

		// Detect values outside [0,1]
		cv::Mat mask_sup = seq_temp > 1.0;
		cv::Mat mask_inf = seq_temp < 0.0;
		mask_sup.convertTo(mask_sup, CV_64F, 1.0 / 255.0);
		mask_inf.convertTo(mask_inf, CV_64F, 1.0 / 255.0);
		// // Clip them
		seq_temp = seq_temp.mul(1.0 - mask_sup) + mask_sup;
		seq_temp = seq_temp.mul(1.0 - mask_inf);
		// cout << "转换前图像序列的最大像素值为：" << getImagePixelMaxValue(seq_temp) << endl;

		saveGeneratedImgSeq(seq_temp, k, dir_path); // 保存生成的图像序列
		saveWeightImgSeqAsHeatMap(wc_temp, k, dir_path);

		// Set to 0 contrast of clipped values
		seq[k + Ns] = seq_temp.clone();
		wc[k + Ns] = wc_temp.clone();
	}

	// Compute well-exposedness weights and final normalized weights
	std::vector<cv::Mat> we(N + Ns + 1);
	std::vector<cv::Mat> w(N + Ns + 1);
	cv::Mat sum_w = cv::Mat::zeros(rows, cols, CV_64F);
	for (int i = 0; i < we.size(); i++)
	{
		cv::Mat p, we_temp, w_temp;
		cv::pow(seq[i] - 0.5, 2, p);
		cv::exp(-0.5 * p / (0.2 * 0.2), we_temp); // 详细版公式8

		w_temp = wc[i].mul(we_temp); // 详细版公式11

		we[i] = we_temp.clone();
		w[i] = w_temp.clone();

		sum_w = sum_w + w[i];
	}

	sum_w = 1.0 / sum_w;
	for (int i = 0; i < we.size(); i++)
	{
		w[i] = w[i].mul(sum_w); // 详细版公式11
	}

	// Multiscale blending
	cv::Mat lp = multiscale_blending(seq, w); // 详细版公式12

	if (channels == 1)
	{
		lp = lp.mul(C);
	}
	else
	{
		std::vector<cv::Mat> lp3 = {lp.clone(), lp.clone(), lp.clone()};
		cv::merge(lp3, lp);
		lp = lp.mul(C);
	}

	robust_normalization(lp, lp);
	cout << "融合结果图像转换成16位归一化之前像素的最大值为：" << getImagePixelMaxValue(lp) << endl;
	double minPVal, maxPVal;
	cv::minMaxLoc(lp, &minPVal, &maxPVal);
	lp = (lp - minPVal) / (maxPVal - minPVal);
	cout << "融合结果图像转换成16位归一化之后像素的最大值为：" << getImagePixelMaxValue(lp) << endl;

	lp.convertTo(dst, CV_16U, 65535);
	cout << "融合结果图像转换成16位之后像素的最大值为：" << getImagePixelMaxValue(dst) << endl;

	string resultImageName = dir_path + "/" +
							 "resultα=" + ToString(alpha) + "，" + "β=" + ToString(beta) + ".tif";
	cv::imwrite(resultImageName, dst); // 保存处理后的图像

	// 将改变参数过程中得到的所有图像保存在result文件夹中
	string all_result_image_name = all_save_path + "/" +
								   "α=" + ToString(alpha) + "，" + "β=" + ToString(beta) + ".tif";
	cv::imwrite(all_result_image_name, dst);
	cout << "融合结果图像保存完成！！！" << endl;

	return;
}

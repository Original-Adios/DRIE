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

// double转string并保留两位小数
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

/*
定义了一个名为laplacian_pyramid的函数，
该函数的返回类型是一个std::vector<cv::Mat>对象，其中cv::Mat是OpenCV中表示图像的类。
函数有两个参数：一个是输入图像src，另一个是金字塔的层数nLevel
*/
std::vector<cv::Mat> laplacian_pyramid(const cv::Mat &src, int nLevel) // 生成拉普拉斯金字塔
{
	cv::Mat I = src.clone();  // 将输入图像src进行复制，生成一个新的cv::Mat对象I。这样做是为了避免在函数中修改原始输入图像。
	std::vector<cv::Mat> pyr; // 定义了一个空的std::vector<cv::Mat>对象pyr
	cv::Mat J = I.clone();	  // 定义了一个新的cv::Mat对象J，它是I的复制品。J将在后面的循环中用作上一级别的图像。
	/*
   这个循环迭代nLevel-1次，每次迭代都生成一层拉普拉斯金字塔。
   在每次迭代中，首先使用cv::pyrDown函数将J下采样到下一级别的图像I。
   然后使用cv::pyrUp函数将I上采样回到原始大小，并将结果存储在J_up中。
   接下来，计算当前层次的差值图像，并将其添加到pyr向量的末尾。
   最后，将J更新为当前层次的图像I，以便在下一次迭代中使用。
   */

	for (int i = 1; i < nLevel; i++)
	{
		cv::pyrDown(J, I); // 降采样 (cv::pyrDown)  J--输入图像（Mat类的对象即可）  I--输出图像
		cv::Mat J_up;
		cv::pyrUp(I, J_up, J.size()); // 上采样(cv::pyrUp)  I--输入图像  J_up--输出图像
		pyr.push_back(J - J_up);
		J = I;
	}
	pyr.push_back(J); // the coarest level contains the residual low pass image
	// 将最后一层金字塔的图像J添加到向量pyr的末尾。这一层包含低通滤波后的图像，也称为残差图像。
	return pyr; // 回包含所有拉普拉斯金字塔图像的向量pyr
}

// 用于重建拉普拉斯金字塔
/*
具体的实现方法是从金字塔的最后一层（即最小分辨率的图像）开始，将其通过 pyrUp 函数上采样到上一层，
然后和上一层的高斯图像相加，得到上一层的图像。这个过程一直持续到金字塔的顶层（即原图像），最后得到的 R 即为重建后的图像。
*/
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
// 用于多尺度融合的函数，输入参数有两个：一个是类型为std::vector<cv::Mat>的图像序列seq，另一个是类型为std::vector<cv::Mat>的权重向量W
cv::Mat multiscale_blending(const std::vector<cv::Mat> &seq, const std::vector<cv::Mat> &W)
{
	int h = seq[0].rows;
	int w = seq[0].cols;
	int n = seq.size();
	// 计算输入图像的参考金字塔层数nScRef。金字塔层数是由输入图像的最小维度（高度和宽度中较小的那个）决定的，通过对它进行对数计算来得到。
	int nScRef = int(std::log(std::min(h, w)) / log(2));

	int nScales = 1; // 金字塔层数nScales
	int hp = h;		 // 每个层次的高度hp
	int wp = w;		 // 每个层次的宽度wp
	while (nScales < nScRef)
	{
		nScales++;
		hp = (hp + 1) / 2;
		wp = (wp + 1) / 2;
	} // 在这个循环中，层数从1开始，高度和宽度根据每次迭代的上一级别计算。迭代将继续，直到达到参考金字塔的层数
	// 这里使用的是自顶向下的金字塔结构，即每个层次的大小都是上一级别大小的一半，因此在低于参考层数的层次上进行插值操作
	//  std::cout << "Number of scales: " << nScales << ", residual's size: " << hp << " x " << wp << std::endl;

	/*
	下面这段代码是多尺度融合函数的主要部分。它使用高斯金字塔和拉普拉斯金字塔对输入图像序列进行处理，
	并将它们与权重向量相乘，得到每个金字塔层次的融合结果。
	最后，使用reconstruct_laplacian_pyramid函数重建拉普拉斯金字塔，得到最终的融合图像。
	*/

	std::vector<cv::Mat> pyr;
	hp = h;
	wp = w;
	/*
	这个循环创建一个名为pyr的空拉普拉斯金字塔向量，其中包含nScales个级别。
	每个级别的大小由上一个级别的大小决定，最初的大小由输入图像的大小决定。
	*/
	for (int scale = 1; scale <= nScales; scale++)
	{
		pyr.push_back(cv::Mat::zeros(hp, wp, CV_64F));
		hp = (hp + 1) / 2;
		wp = (wp + 1) / 2;
	}
	/*
	这个循环迭代n次，每次迭代将输入图像序列中的一个图像转换为拉普拉斯金字塔和一个对应的高斯金字塔。
	然后，它将每个金字塔层次的权重与对应的拉普拉斯金字塔层次相乘，并将它们累加到pyr向量的相应级别中。
	这个过程得到了每个金字塔层次的融合结果
	*/

	// 从i=1开始融合，不使用最暗的那张
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
	// 调用reconstruct_laplacian_pyramid函数，将融合结果向量pyr作为输入，重建原始图像金字塔并返回最终的融合结果。
}
// 鲁棒归一化
void robust_normalization(const cv::Mat &src, cv::Mat &dst, double wSat = 1.0, double bSat = 1.0)
{
	int H = src.rows;		// 输入图像的高
	int W = src.cols;		// 输入图像的宽
	int D = src.channels(); // 输入图像的通道数
	int N = H * W;			// 像素总数
	/*
	这个代码块计算输入图像的最大值和最小值。如果输入图像是彩色图像，它将每个通道的最大值和最小值分别计算，
	并使用它们的最大值和最小值作为输入图像的最大值和最小值。
	如果输入图像是灰度图像，它将计算整个图像的最大值和最小值。wSat和bSat参数用于控制鲁棒归一化的饱和度，它们是以百分比形式给出的。
	*/
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
		/*
		这段代码是用于对图像进行阈值处理的。以下是对每行代码的详细解释：

1. 首先，根据最大值vmax和最小值vmin，创建两个二值掩模mask_over和mask_below。其中，mask_over用于标记像素值大于vmax的像素，mask_below用于标记像素值小于vmin的像素。

2. 接着，将mask_over和mask_below转换为双精度浮点数类型CV_64F，并除以255，使像素值在0到1之间。

3. 然后，对原始图像dst进行处理。首先，将dst中mask_over为1的部分设置为1，即将超出阈值范围的像素值都设置为vmax。
这里使用了OpenCV中的mul函数，将掩模和原始图像相乘，对于mask_over为0的像素，相乘结果为0，对于mask_over为1的像素，相乘结果不变，达到了将超过阈值的像素值设置为vmax的效果。

4. 最后，将dst中mask_below为1的部分设置为0，即将低于阈值范围的像素值都设置为0。
同样使用了mul函数，将掩模和原始图像相乘，对于mask_below为1的像素，相乘结果为0，对于mask_below为0的像素，相乘结果不变，达到了将低于阈值的像素值设置为0的效果。

综上，这段代码的作用是将图像中超过阈值的像素值设置为vmax，将低于阈值的像素值设置为0。
		*/
	}

	return;
	/*
	如果vmax和vmin相等，说明图像中所有像素值都相同，此时直接将输出图像设为全为vmax的图像。
	否则，将图像范围归一化到[0,1]。
	如果输入图像是彩色图像，将对每个通道都进行归一化，使用cv::mul函数将其与1.0 / (vmax3 - vmin3)相乘。

接下来，使用mask_over和mask_below来处理超出范围的像素值。具体来说，将dst中大于vmax的像素值设为1，
小于vmin的像素值设为0，并使用cv::convertTo函数将它们转换为double类型。
然后使用cv::mul函数将其与Ones - mask_over相乘，将超出范围的像素值设为vmax。
最后，使用cv::mul函数将其与Ones - mask_below相乘，将小于vmin的像素值设为0。
	*/
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
	/*
	以上这个代码块定义了名为L的变量，它将存储输入图像的亮度通道。如果输入图像是灰度图像，那么直接将它复制到L中；
	否则，将输入图像转换为HSV色彩空间，并将HSV通道分离成一个名为HSV_channels的向量。然后，将亮度通道存储在L中。
	*/
	cv::Mat L_norm; // 把原图像的所有像素压缩到0~1
	L.convertTo(L_norm, CV_64F, 1.0 / 65535.0);
	cv::Mat src_norm;
	src.convertTo(src_norm, CV_64F, 1.0 / 65535.0);
	cout << "原图像归一化后的信息：" << endl;
	// 以上将L和输入图像归一化到0到1之间，并将它们存储在L_norm和src_norm中
	/*像素值在0~1之间的图像通常被称为归一化图像。这种图像是经过处理，将像素值缩放到0和1之间，
	使得每个像素都可以表示为一个相对亮度或颜色强度的比例。
	这种处理通常用于图像处理和机器学习应用中，如深度学习中的图像分类和识别。
	*/
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
	/*以上这个代码块计算SEF增强滤波器的输入。如果输入图像是灰度图像，则将C设置为输入图像除以L_norm加上一个很小的数（2^-16）
	如果输入图像是彩色图像，则先计算一个大小为3的数组，其中每个元素均为输入图像除以L_norm加上一个很小的数，然后将这个数组合并成一个3通道的图像。
	最后，将C设置为输入图像乘以这个数组。这个C变量将在后面的步骤中用于计算SEF增强滤波器的输出。
	*/
	/*
	首先判断原图像的通道数，如果是单通道图像，则将原图像除以亮度通道L加上一个较小的常数（这里是2的-16次方），
	然后将结果赋给对比度增强后的图像C。如果是RGB图像，则首先计算出L的归一化倒数，然后将其复制到三个通道，
	最后将原图像与这个三通道矩阵逐像素相乘得到对比度增强后的图像C。
	*/
	/*
	这个过程的原理是：对图像进行除法操作相当于对其进行了一次归一化，可以将图像的像素值缩放到[0, 1]之间；
	而加上一个常数可以使得图像中的小值更容易被拉升，增强图像的对比度。
	（除以1个1~0之间的数相当于增大那个数的值）
	*/
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

	/*
	while循环用于估计生成图像的数量。如果t_max和t_min的范围交叉，则停止迭代。
	在每次迭代中，代码都会更新生成图像的数量、t_min和t_max的值，以及用于计算最大因子的Nx值。
	如果估计的图像数量超过了49个，则输出错误信息并停止迭代。
	*/
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
	// fun_r是一个用于计算动态范围缩放偏移的函数，它接受一个整数int k作为参数。
	// beta、Ns和Mp变量用于计算偏移量。这个函数返回一个偏移量，用于减小输入图像的动态范围。

	// Reduce dynamic (using offset function "r")
	// fun_g是用于动态范围缩放的函数，它接受一个像素值cv::Mat t和一个整数int k作为参数。
	// 这个函数使用了fun_r计算出的偏移量来缩放输入图像的动态范围。a、b和lambda变量用于计算缩放系数。这个函数返回一个缩放后的像素值。
	double a = beta / 2 + lambda; // 公式8下方的解释
	double b = beta / 2 - lambda;
	auto fun_g = [fun_r, beta, a, b, lambda](cv::Mat t, int k)
	{
		auto rk = fun_r(k);
		cv::Mat diff = t - rk;
		cv::Mat abs_diff = cv::abs(diff);

		cv::Mat mask = abs_diff <= beta / 2; // mask用来标记亮度没有超过范围的像素位置
		// cout << "mask图像的像素值类型：" << mask.type() << endl;
		// std::cout << "mask最大像素值为: " << getImagePixelMaxValue(mask) << std::endl;
		// cout << "归一化前mask图像的信息：" << endl;
		// printImgInformation(mask);

		mask.convertTo(mask, CV_64F, 1.0 / 255.0);
		// cout << "归一化后mask图像的信息：" << endl;
		// printImgInformation(mask);
		// std::cout << "转换后mask最大像素值为: " << getImagePixelMaxValue(mask) << std::endl;

		cv::Mat sign = diff.mul(1.0 / abs_diff);

		return mask.mul(t) + (1.0 - mask).mul(sign.mul(a - lambda * lambda / (abs_diff - b)) + rk); // 公式8
	};
	/*
	公式8的解释
	*/
	// final remapping functions: h = g o f
	// fun_h和fun_hs分别是用于创建更亮和更暗的图像的最终映射函数，它们是通过将fun_f或fun_fs作为输入传递给fun_g得到的
	auto fun_h = [fun_f, fun_g](cv::Mat t, int k) { // create brighter images (k>=0) (enhance dark parts)
		return fun_g(fun_f(t, k), k);
	};
	auto fun_hs = [fun_fs, fun_g](cv::Mat t, int k) { // create darker images (k<0) (enhance bright parts)
		return fun_g(fun_fs(t, k), k);
	};

	// derivative of g with respect to t
	// fun_dg是对fun_g函数相对于输入图像像素值t的导数函数,这个函数返回一个与输入图像形状相同的矩阵，表示每个像素的导数。
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

	// derivative of the remapping functions: dh = f' x g' o f
	// fun_dh和fun_dhs分别是fun_h和fun_hs函数相对于输入图像像素值t的导数函数
	auto fun_dh = [alpha, Nx, fun_f, fun_dg](cv::Mat t, int k)
	{
		return std::pow(alpha, k * 1.0 / Nx) * fun_dg(fun_f(t, k), k);
	};
	auto fun_dhs = [alpha, Nx, fun_fs, fun_dg](cv::Mat t, int k)
	{
		return std::pow(alpha, -k * 1.0 / Nx) * fun_dg(fun_fs(t, k), k);
	};

	// Simulate a sequence from image L_norm and compute the contrast weights
	// 这段代码模拟了从输入图像L_norm中生成的一组序列，并计算每个序列的对比度权重。seq和wc分别是图像序列和对应的对比度权重序列。
	std::vector<cv::Mat> seq(N + Ns + 1);
	std::vector<cv::Mat> wc(N + Ns + 1);
	/*
	在循环中，每个序列都是通过将输入图像L_norm传递给fun_h或fun_hs函数来应用最终映射函数生成的。
	然后，通过将L_norm传递给fun_dh或fun_dhs函数来计算对应的对比度权重。
	在计算完成后，检测序列中的值是否超出了[0,1]的范围，并将其进行裁剪。如果裁剪了某个像素，则将对应的对比度权重设置为0。
	最后，将生成的序列和对比度权重序列存储在seq和wc向量中。
	*/
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
		// 保存生成的权重图
		/*生成的权重图通常是单通道的灰度图像。
		热力图是一种常见的可视化方法，它将权重图表示为伪彩色图像以更直观地展示权重的分布情况。
		热力图通常是三通道的，其中每个像素的颜色值由权重图对应像素的灰度值映射而来。
		OpenCV中的applyColorMap函数可以将灰度图像转换为伪彩色图像，输出的图像为三通道的图像。
		当我们保存热力图时，保存的是经过applyColorMap函数转换后的三通道伪彩色图像，而不是原始的单通道权重图。

		在OpenCV中，可以使用applyColorMap函数将灰度图像转换为伪彩色图像。
		此函数接受两个参数：输入灰度图像和输出彩色图像。它会根据灰度图像的像素值，
		根据所选的颜色映射表，将每个像素的灰度值映射为相应的伪彩色值。

		例如，使用JET颜色映射表，较低的灰度值会映射为蓝色，中等值映射为绿色，较高值映射为红色。
		RAINBOW颜色映射表会在不同的灰度值之间产生彩虹色彩渐变。
		HOT颜色映射表会从黑色过渡到红色，再过渡到白色，用于表示温度或热度。
		*/
		saveWeightImgSeqAsHeatMap(wc_temp, k, dir_path);

		// Set to 0 contrast of clipped values
		// wc_temp = wc_temp.mul(1.0 - mask_sup); // 将强度值在0~1之外的像素位置的权重设为0
		// wc_temp = wc_temp.mul(1.0 - mask_inf);

		seq[k + Ns] = seq_temp.clone();
		wc[k + Ns] = wc_temp.clone();
	}

	// Compute well-exposedness weights and final normalized weights
	// 这段代码计算了序列的曝光程度权重和最终权重，以便为每个像素选择最优的像素值。we和w分别是序列的曝光程度权重和最终权重序列。
	std::vector<cv::Mat> we(N + Ns + 1);
	std::vector<cv::Mat> w(N + Ns + 1);
	cv::Mat sum_w = cv::Mat::zeros(rows, cols, CV_64F);
	/*
	在循环中，we_temp是通过计算序列seq与0.5之间的距离的平方，并将其作为指数函数的指数来计算的。
	这个计算的结果表示每个像素的曝光程度权重。然后，将对比度权重wc与曝光程度权重相乘得到最终权重w_temp。
	*/
	// 在计算完成后，将曝光程度权重和最终权重分别存储在we和w向量中。最后，对所有权重进行归一化，使它们的总和为1。这里使用了sum_w的倒数来实现归一化。
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
	// 这段代码使用多尺度融合算法对序列seq进行融合，生成最终的图像输出。
	// multiscale_blending函数是一个自定义函数，用于执行多尺度融合。该函数接受序列seq和对应的最终权重w作为输入，并返回融合后的图像。
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

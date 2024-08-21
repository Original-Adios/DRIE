#ifndef _IMAGE_ENHANCEMENT_H
#define _IMAGE_ENHANCEMENT_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
void SEF(const cv::Mat &src, cv::Mat &dst, double alpha = 6.0, double beta = 0.5, double lambda = 0.125);
double getImagePixelMaxValue(cv::Mat img);                      // 获取一幅图像的最大像素值；
void saveGeneratedImgSeq(cv::Mat img, int i, string save_path); // 保存生成的图像序列
void printImgInformation(cv::Mat img);                          // 打印图像信息
#endif

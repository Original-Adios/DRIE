#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace std;
#include "image_enhancement.h"

using std::chrono::high_resolution_clock;

void MyTimeOutput(const std::string &str, const high_resolution_clock::time_point &start_time, const high_resolution_clock::time_point &end_time)
{
    std::cout << str << std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0 << "ms" << std::endl;
    return;
}

int main(int argc, char **argv)
{
    cv::Mat src = cv::imread(argv[1], -1);
    printImgInformation(src);
    if (src.empty())
    {
        std::cout << "Can't read image file." << std::endl;
        return -1;
    }
    // cv::line(src, cv::Point(390, 0), cv::Point(390, src.rows), 65535, 2);
    if (src.type() == 16)
    {
        cv::Mat gray;
        cvtColor(src, gray, cv::COLOR_RGB2GRAY);
        gray.convertTo(gray, CV_16U, 65535 / 255);
        printImgInformation(gray);
        high_resolution_clock::time_point start_time, end_time;
        cv::Mat SEF_dst;
        start_time = high_resolution_clock::now();
        SEF(gray, SEF_dst, atof(argv[2]), atof(argv[3]));
        end_time = high_resolution_clock::now();
        MyTimeOutput("SEF处理时间: ", start_time, end_time);
    }
    else
    {
        high_resolution_clock::time_point start_time, end_time;
        cv::Mat SEF_dst;
        // α：每隔2增长
        // β：每隔0.1增长
        for (double alpha = atof(argv[2]); alpha < atof(argv[3]); alpha += 1)
        {
            for (double beta = atof(argv[4]); beta < atof(argv[5]); beta += 0.1)
            {
                start_time = high_resolution_clock::now();
                SEF(src, SEF_dst, alpha, beta);
                end_time = high_resolution_clock::now();
                MyTimeOutput("SEF处理时间: ", start_time, end_time);
            }
        }

        // start_time = high_resolution_clock::now();
        // SEF(src, SEF_dst, atof(argv[2]), atof(argv[3]));
        // end_time = high_resolution_clock::now();
        // MyTimeOutput("SEF处理时间: ", start_time, end_time);
    }

    // cv::imwrite("result.tif", SEF_dst); // 保存处理后的图像
    cv::waitKey();
    return 0;
}
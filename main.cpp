#include<iostream>
#include<opencv2/opencv.hpp>
#include "SIFT.h"
using namespace cv;
using namespace std;

int main()
{
    Mat img = imread("/home/sahil/demo/lena.jpg");
    SIFT x(img,4,2);
    x.startSift();
    x.ShowKeypoints();
    return 0;
}

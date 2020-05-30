#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "keypoint.h"
#include "descriptor.h"
using namespace cv;
class SIFT
{
private:
    vector<vector<Mat> >m_glist;    // A 2D array to hold the different gaussian blurred images
    vector<vector<Mat> >m_dogList;   // A 2D array to hold the different DoG images
    vector<vector<Mat> >m_extrema;    // A 2D array to hold binary images. In the binary image, 1 = extrema, 0 = not extrema
    vector<vector<double> >m_absSigma;		// A 2D array to hold the sigma used to blur a particular image
    vector<Keypoint> m_keyPoints;	// Holds each keypoint's basic info
    vector<Descriptor> m_keyDescs;	// Holds each keypoint's descriptor
    Mat srcImage;			// The image we're working on
    int m_numOctaves;		// The desired number of octaves
    int m_numIntervals;	// The desired number of intervals
    int m_numKeypoints;	// The number of keypoints detected

    //void GenerateLists();
    void BuildScaleSpace();
    void DetectExtrema();
    void AssignOrientations();
    void ExtractKeypointDescriptors();
    int GetKernelSize(double sigma, double cut_off = 0.001);
    Mat BuildInterpolatedGaussianTable(int size, double sigma);
    double gaussian2D(double x, double y, double sigma);

public:
    SIFT(Mat img, int octaves, int intervals);

    void startSift();
    void ShowKeypoints();
    //void ShowAbsSigma();

};

#include "SIFT.h"
#include "stdafx.h"
#include <climits>

#define SIGMA_ANTIALIAS			0.5
#define SIGMA_PREBLUR			1.0
#define CURVATURE_THRESHOLD		5.0
#define CONTRAST_THRESHOLD		0.03		// in terms of 255
#define M_PI					3.1415926535897932384626433832795
#define NUM_BINS				36
#define MAX_KERNEL_SIZE			20
#define FEATURE_WINDOW_SIZE		16
#define DESC_NUM_BINS			8
#define FVSIZE					128
#define	FV_THRESHOLD			0.2

using namespace cv;

//intialising the the meSIFT class
//with a constructor,taking 3 arguments
//the src image, the number of octaves in the image
//and the number of intervals
SIFT::SIFT(Mat src,int octaves,int intervals)
{
    srcImage = src.clone();
    m_numOctaves = octaves;
    m_numIntervals = intervals;
}

// DoSift()
// This function does everything in sequence.
void SIFT::startSift()
{
    BuildScaleSpace();
    DetectExtrema();
    AssignOrientations();
    //ExtractKeypointDescriptors();
    return;
}

// BuildScaleSpace()
// This function generates all the blurred out images for each octave
// and also the DoG images
void SIFT::BuildScaleSpace()
{

    Mat src_gray;
    //intialising a floating point image
    Mat imgFloat;
    //we need a gray scale image
    cvtColor(srcImage,src_gray, COLOR_BGR2GRAY);
    //we set each and every pixel of the floating point image
    //converting from range(0 to 255) to (0 to 1)
    src_gray.convertTo(imgFloat,CV_32F,1.0/255,0);
;


    GaussianBlur(imgFloat, imgFloat, Size(0, 0), SIGMA_ANTIALIAS);               //CHANGEABLE
    resize(imgFloat,imgFloat,Size(imgFloat.size()*2));


    m_glist.push_back(vector<Mat>());
    m_glist[0].push_back(imgFloat.clone());
    pyrUp(imgFloat, m_glist[0][0]);

    GaussianBlur(m_glist[0][0],m_glist[0][0],Size(0,0),SIGMA_PREBLUR);


    //keeping a track of the sigmas
    double initSigma = sqrt(2.0f);
    m_absSigma.push_back(vector<double>());
    m_absSigma[0].push_back(initSigma * 0.5);

    //now for the actual image generation
    //we run this 2D loop
    int i, j;
    for (i = 0; i < m_numOctaves; i++)
    {
        // Reset sigma for each octave
        double sigma = initSigma;
        Size s = m_glist[i][0].size();
        m_dogList.push_back(vector<Mat>());

        for (j = 1; j<int(m_numIntervals+3); j++)
        {
            // Calculate a sigma to blur the current image to get the next one
            double sigma_f = sqrt(pow(2.0, 2.0 / m_numIntervals) - 1) * sigma;
            sigma = pow(2.0, 1.0 / m_numIntervals) * sigma;
            // Store sigma values (to be used later on)
            m_absSigma[i].push_back(sigma * 0.5 * pow(2.0f, (float)i));

            //applying Gaussian blur
            Mat temp;
            GaussianBlur(m_glist[i][j-1], temp, Size(0, 0), sigma_f);
            m_glist[i].push_back(temp);

            //finding and allocating the DoG image using the vector
            Mat temp2;
            subtract(m_glist[i][j-1],m_glist[i][j],temp2,Mat());
            m_dogList[i].push_back(temp2);

        }
        if (i < m_numOctaves - 1)
        {
            Mat temp3;
            pyrDown(m_glist[i][0], temp3, s/2);
            m_absSigma.push_back(vector<double>());
            m_glist.push_back(vector<Mat>());
            m_glist[i+1].push_back(temp3);
            m_absSigma[i+1].push_back(m_absSigma[i][m_numIntervals]);
        }
    }
}

// DetectExtrema()
// Locates extreme points (maxima and minima)
void SIFT::DetectExtrema()
{
    double curvature_ratio, curvature_threshold;
    int scale;
    double dxx, dyy, dxy, trH, detH;
    unsigned int num = 0;				// Number of keypoints detected
    unsigned int numRemoved = 0;		// The number of key points rejected
    Mat middle, up, down;
    curvature_threshold = (CURVATURE_THRESHOLD + 1) * (CURVATURE_THRESHOLD + 1) / CURVATURE_THRESHOLD;

    //Detect extrema in the DoG image
    //using almost four loops
    for (int i = 0; i < m_numOctaves; i++)
    {
        scale= (int)pow(2.0, (double)i);
        m_extrema.push_back(vector<Mat>());
        for (int j = 1; j < m_numIntervals + 1; j++)
        {
            // Allocate memory and set all points to zero ("not key point")
            Mat temp = Mat::zeros(m_dogList[i][0].size(), CV_8UC1);
            m_extrema[i].push_back(temp);

            //finding the images just above and below
            //for the current octave
            middle = m_dogList[i][j];
            up = m_dogList[i][j + 1];
            down = m_dogList[i][j - 1];

            //intializing certain variables
            int xi, yi;
            for (xi = 1; xi < m_dogList[i][j].rows - 1; xi++)
            {
                for (yi = 1; yi < m_dogList[i][j].cols - 1; yi++)
                {
                    // true if a keypoint is a maxima/minima
                    // but needs to be tested for contrast/edge thingy
                    bool justSet = false;
                    float curr = middle.at<float>(Point(yi, xi));
                    if (curr > middle.at<float>(Point(yi + 1, xi)) &&
                        curr > middle.at<float>(Point(yi - 1, xi)) &&
                        curr > middle.at<float>(Point(yi, xi + 1)) &&
                        curr > middle.at<float>(Point(yi, xi - 1)) &&
                        curr > middle.at<float>(Point(yi + 1, xi + 1)) &&
                        curr > middle.at<float>(Point(yi - 1, xi - 1)) &&
                        curr > middle.at<float>(Point(yi - 1, xi + 1)) &&
                        curr > middle.at<float>(Point(yi + 1, xi - 1)) &&
                        curr > up.at<float>(Point(yi, xi)) &&
                        curr > up.at<float>(Point(yi + 1, xi)) &&
                        curr > up.at<float>(Point(yi - 1, xi)) &&
                        curr > up.at<float>(Point(yi, xi + 1)) &&
                        curr > up.at<float>(Point(yi, xi - 1)) &&
                        curr > up.at<float>(Point(yi + 1, xi + 1)) &&
                        curr > up.at<float>(Point(yi - 1, xi - 1)) &&
                        curr > up.at<float>(Point(yi - 1, xi + 1)) &&
                        curr > up.at<float>(Point(yi + 1, xi - 1)) &&
                        curr > down.at<float>(Point(yi, xi)) &&
                        curr > down.at<float>(Point(yi + 1, xi)) &&
                        curr > down.at<float>(Point(yi - 1, xi)) &&
                        curr > down.at<float>(Point(yi, xi + 1)) &&
                        curr > down.at<float>(Point(yi, xi - 1)) &&
                        curr > down.at<float>(Point(yi + 1, xi + 1)) &&
                        curr > down.at<float>(Point(yi - 1, xi - 1)) &&
                        curr > down.at<float>(Point(yi - 1, xi + 1)) &&
                        curr > down.at<float>(Point(yi + 1, xi - 1))
                        )
                    {
                        num++;
                        justSet = true;
                        m_extrema[i][j - 1].at<uchar>(Point(yi, xi)) = 255;
                    }

                    //checking if it's a minima
                    if (curr < middle.at<float>(Point(yi + 1, xi)) &&
                        curr < middle.at<float>(Point(yi - 1, xi)) &&
                        curr < middle.at<float>(Point(yi, xi + 1)) &&
                        curr < middle.at<float>(Point(yi, xi - 1)) &&
                        curr < middle.at<float>(Point(yi + 1, xi + 1)) &&
                        curr < middle.at<float>(Point(yi - 1, xi - 1)) &&
                        curr < middle.at<float>(Point(yi - 1, xi + 1)) &&
                        curr < middle.at<float>(Point(yi + 1, xi - 1)) &&
                        curr < up.at<float>(Point(yi, xi)) &&
                        curr < up.at<float>(Point(yi + 1, xi)) &&
                        curr < up.at<float>(Point(yi - 1, xi)) &&
                        curr < up.at<float>(Point(yi, xi + 1)) &&
                        curr < up.at<float>(Point(yi, xi - 1)) &&
                        curr < up.at<float>(Point(yi + 1, xi + 1)) &&
                        curr < up.at<float>(Point(yi - 1, xi - 1)) &&
                        curr < up.at<float>(Point(yi - 1, xi + 1)) &&
                        curr < up.at<float>(Point(yi + 1, xi - 1)) &&
                        curr < down.at<float>(Point(yi, xi)) &&
                        curr < down.at<float>(Point(yi + 1, xi)) &&
                        curr < down.at<float>(Point(yi - 1, xi)) &&
                        curr < down.at<float>(Point(yi, xi + 1)) &&
                        curr < down.at<float>(Point(yi, xi - 1)) &&
                        curr < down.at<float>(Point(yi + 1, xi + 1)) &&
                        curr < down.at<float>(Point(yi - 1, xi - 1)) &&
                        curr < down.at<float>(Point(yi - 1, xi + 1)) &&
                        curr < down.at<float>(Point(yi + 1, xi - 1))
                        )
                    {
                        num++;
                        justSet = true;
                        m_extrema[i][j - 1].at<uchar>(Point(yi, xi)) = 255;
                    }

                    //contrast check
                    if (justSet && fabs(middle.at<float>(Point(yi, xi))) < CONTRAST_THRESHOLD)
                    {
                        num--;
                        numRemoved++;
                        m_extrema[i][j - 1].at<uchar>(Point(yi, xi)) = 0;
                        justSet = false;
                    }
                    //edge check
                    if (justSet)
                    {
                        dxx = (middle.at<float>(Point(yi + 1, xi)) + middle.at<float>(Point(yi - 1, xi)) - 2.0 * middle.at<float>(Point(yi, xi)));
                        dyy = (middle.at<float>(Point(yi, xi + 1)) + middle.at<float>(Point(yi, xi - 1)) - 2.0 * middle.at<float>(Point(yi, xi)));
                        dxy = (middle.at<float>(Point(yi + 1, xi + 1)) + middle.at<float>(Point(yi - 1, xi - 1)) - middle.at<float>(Point(yi - 1, xi + 1)) - middle.at<float>(Point(yi + 1, xi - 1))) / 4.0;


                        trH = dxx + dyy;
                        detH = dxx * dyy - dxy * dxy;

                        curvature_ratio = trH * trH / detH;
                        //printf("Threshold: %f - Ratio: %f\n", curvature_threshold, curvature_ratio);
                        if (detH<0 || curvature_ratio>curvature_threshold)
                        {
                            num--;
                            numRemoved++;
                            m_extrema[i][j - 1].at<uchar>(Point(yi, xi)) = 0;
                            justSet = false;
                        }
                    }
                }
            }


        }
    }

    m_numKeypoints = num;
    //printf("Found %d keypoints\n", num);
    //printf("Rejected %d keypoints\n", numRemoved);
}

// GetKernelSize()
// Returns the size of the kernal for the Gaussian blur given the sigma and
// cutoff value.
int SIFT::GetKernelSize(double sigma, double cut_off)
{
    int i;
    for (i = 0; i < MAX_KERNEL_SIZE; i++)
        if (exp(-((double)(i * i)) / (2.0 * sigma * sigma)) < cut_off)
            break;
    int size = 2 * i - 1;
    return size;
}


// AssignOrientations()
// For all the key points, generate an orientation.
void SIFT::AssignOrientations()
{
    int i, j, k, xi, yi;
    int kk, tt;

    //the 2-D vectors for Magnitude and orientation
    vector<vector<Mat> >magnitude;
    vector<vector<Mat> >orientation;

    //iterate through all the octaves
    for (i = 0; i < m_numOctaves; i++)
    {
        magnitude.push_back(vector<Mat>());
        orientation.push_back(vector<Mat>());
        for (j = 1; j < m_numIntervals + 1; j++)
        {
            Mat temp1 = Mat::zeros(m_glist[i][j].size(),CV_32FC1);
            Mat temp2 = Mat::zeros(m_glist[i][j].size(), CV_32FC1);
            magnitude[i].push_back(temp1);
            orientation[i].push_back(temp2);

            // Iterate over the gaussian image with the current octave and interval
            for (xi = 1; xi < m_glist[i][j].rows-1; xi++)
            {
                for (yi = 1; yi < m_glist[i][j].cols-1; yi++)
                {
                    float dx = m_glist[i][j].at<float>(Point(yi, xi + 1)) - m_glist[i][j].at<float>(Point(yi, xi - 1));
                    float dy=  m_glist[i][j].at<float>(Point(yi+1, xi)) - m_glist[i][j].at<float>(Point(yi-1, xi));

                    //storing the magnitude in the new magnitude grid
                    magnitude[i][j - 1].at<float>(Point(yi, xi)) = sqrt(dx*dx+dy*dy);

                    //calculating and storing the orientation
                    float ori = atan(dy / dx);
                    orientation[i][j - 1].at<float>(Point(yi, xi)) = ori;
                }
            }
            //imshow(to_string(10 * j + i), magnitude[i][j-1]);
            //waitKey(0);
            //imshow(to_string(10 * j + i), orientation[i][j-1]);
            //waitKey(0);
        }
    }

    // The histogram with 8 bins
    float* hist_orient = new float[NUM_BINS];
    //iterate thru all the octaves
    for (int i = 0; i < m_numOctaves; i++)
    {
        int scale = int(pow(2.0,double(i)));
        int width = m_glist[i][0].cols;
        int height = m_glist[i][0].rows;
        //iterate thru all the intervals in the current scale
        for (int j = 1; j < m_numIntervals + 1; j++)
        {
            double abs_sigma = m_absSigma[i][j];
            Mat img_weight=Mat::zeros(magnitude[i][j-1].size(),CV_32FC1);
            GaussianBlur(magnitude[i][j-1], img_weight, Size(0, 0), 1.5 * abs_sigma);

            // Get the kernel size for the Guassian blur
            int h = SIFT::GetKernelSize(1.5 * abs_sigma) / 2;
            //temporarily creating a mask of region
            //to calculate the orientation
            Mat img_mask = Mat::zeros(Size(width,height),CV_8UC1);

            //iterate thru all the points in the octave
            //and the interval
            for (xi = 1; xi < height-1; xi++)
            {
                for (yi = 1; yi < width-1; yi++)
                {
                    int val = (int)m_extrema[i][j - 1].at<uchar>(Point(yi, xi));
                    if (val!=0)
                    {
                        // Reset the histogram thingy
                        for (k = 0; k < NUM_BINS; k++)
                            hist_orient[k] = 0.0;
                        // Go through all pixels in the window around the extrema
                        for (kk = -h; kk <= h; kk++)
                        {
                            for (tt = -h; tt <= h; tt++)
                            {
                                // Ensure we're within the image
                                if (xi + kk < 0 || xi + kk >= height || yi + tt < 0 || yi + tt >= width)
                                    continue;

                                float sample_orient = orientation[i][j-1].at<float>(Point(yi + tt, xi + kk));

                                //if (sampleOrient <= -M_PI || sampleOrient > M_PI)
                                    //printf("Bad Orientation: %f\n", sampleOrient);      //CHANGEABLE

                                sample_orient+=(float)M_PI;

                                int sample_orient_degrees = int(sample_orient * 180 / M_PI);
                                hist_orient[(int)sample_orient_degrees / (360 / NUM_BINS)] += img_weight.at<float>(Point(yi + tt, xi + kk));
                                img_mask.at<uchar>(Point(yi + tt, xi + kk))=255;

                            }
                        }

                        // We've computed the histogram. Now check for the maximum
                        float max_peak = hist_orient[0];
                        int max_peak_index = 0;
                        for (k = 1; k < NUM_BINS; k++)
                        {
                            if (hist_orient[k] > max_peak)
                            {
                                max_peak = hist_orient[k];
                                max_peak_index = k;
                            }
                        }

                        //list of magnitudes and orientation at the current extrema
                        vector<float> orien;
                        vector<float> mag;

                        for (int k = 0; k < NUM_BINS; k++)
                        {
                            // Do we have a good peak?
                            if (hist_orient[k] > 0.8 * max_peak)
                            {
                                // Three points. (x2,y2) is the peak and (x1,y1)
                                // and (x3,y3) are the neigbours to the left and right.
                                // If the peak occurs at the extreme left, the "left
                                // neighbour" is equal to the right most. Similarly for
                                // the other case (peak is rightmost)
                                float x1 = (float)(k - 1);
                                float y1;
                                float x2 = (float)k;
                                float y2 = hist_orient[k];
                                float x3 = (float)(k + 1);
                                float y3;
                                //cout << x1 << " " << x2 << " " << x3 << endl;
                                if  (k ==0)
                                {
                                    y1 = hist_orient[NUM_BINS - 1];
                                    y3 = hist_orient[1];
                                }
                                else if (k == NUM_BINS - 1)
                                {
                                    y1 = hist_orient[NUM_BINS - 2];             ///CHANGED
                                    y3 = hist_orient[0];
                                }
                                else
                                {
                                    y1 = hist_orient[k - 1];
                                    y3 = hist_orient[k + 1];
                                }



                                // Next we fit a downward parabola aound
                                // these three points for better accuracy

                                // A downward parabola has the general form
                                //
                                // y = a * x^2 + bx + c
                                // Now the three equations stem from the three points
                                // (x1,y1) (x2,y2) (x3.y3) are
                                //
                                // y1 = a * x1^2 + b * x1 + c
                                // y2 = a * x2^2 + b * x2 + c
                                // y3 = a * x3^2 + b * x3 + c
                                //
                                // in Matrix notation, this is y = Xb, where
                                // y = (y1 y2 y3)' b = (a b c)' and
                                //
                                //     x1^2 x1 1
                                // X = x2^2 x2 1
                                //     x3^2 x3 1
                                //
                                // OK, we need to solve this equation for b
                                // this is done by i//imshow(to_string(10*j+i),m_extrema[i][j-1]);
                                //waitKey(0);nverse the matrix X
                                //
                                // b = inv(X) Y

                                float* b = new float[3];
                                Mat X = Mat(Size(3,3),CV_32FC1);
                                Mat matInv = Mat(Size(3,3),CV_32FC1);
                                X.at<float>(Point(0,0))=x1*x1;
                                X.at<float>(Point(1,0))=x1;
                                X.at<float>(Point(2,0))=1;
                                X.at<float>(Point(0,1))=x2*x2;
                                X.at<float>(Point(1,1))=x2;
                                X.at<float>(Point(2,1))=1;
                                X.at<float>(Point(0,2))=x3*x3;
                                X.at<float>(Point(1,2))=x3;
                                X.at<float>(Point(2, 2)) =1;

                                //finding and storing the inverse
                                invert(X,matInv);
                                /*for (int i = 0; i < 3; i++)
                                {
                                    for (int j = 0; j < 3; j++)
                                    {
                                        cout << matInv.at<float>(Point(j,i)) << " ";
                                    }
                                    cout << endl;
                                }*/

                                b[0] = matInv.at<float>(Point(0,0))*y1+matInv.at<float>(Point(1,0))*y2+matInv.at<float>(Point(2,0))*y3;
                                b[1] = matInv.at<float>(Point(0,1))*y1+matInv.at<float>(Point(1,1))*y2+matInv.at<float>(Point(2,1))*y3;
                                b[2] = matInv.at<float>(Point(0, 2))*y1 + matInv.at<float>(Point(1, 2))*y2 + matInv.at<float>(Point(2, 2))*y3;

                                float x0 = -b[1] / (2 * b[0]);
                                //
                                //cout << b[0] << endl;
                                // Anomalous situation
                                if (fabs(x0) > 2 * NUM_BINS)
                                    x0 = x2;
                                while (x0 < 0)
                                    x0 += NUM_BINS;
                                while (x0 >= NUM_BINS)
                                    x0 -= NUM_BINS;

                                // Normalize it
                                float s = (float)M_PI / NUM_BINS;
                                float x0_n = x0 * (2 * s);
                                //cout << x0_n << endl;
                                assert(x0_n >= 0 && x0_n < 2 * M_PI);
                                x0_n -= (float)M_PI;
                                assert(x0_n >= -M_PI && x0_n < M_PI);

                                orien.push_back(x0_n);
                                //cout << orien.back() << " ";
                                mag.push_back(hist_orient[k]);



                            }
                        }
                        //cout << endl;
                        // Save this keypoint into the list
                        float r = float(xi * scale / 2);
                        float s = float(yi * scale / 2);
                        //cout << orien[0] << endl;
                        //if (orien.empty()) cout << "EMpty!" << endl;
                        Keypoint k(r, s, mag, orien, i* m_numIntervals + j - 1);
                        m_keyPoints.push_back(k);
                    }
                }
            }

            //imshow(to_string(10 * j + i), img_mask);
            //waitKey(0);
        }
    }

    // Finally, we're done with all the magnitude and orientation images.
    //assert(m_keyPoints.size() == m_numKeypoints);

}



// BuildInterpolatedGaussianTable()
// This function actually generates the bell curve like image for the weighted
// addition earlier.
Mat SIFT::BuildInterpolatedGaussianTable(int size, double sigma)
{
    int i, j;
    double half_kernel_size = size / 2 - 0.5;

    double sog = 0;
    Mat ret=Mat::zeros(size,size,CV_32FC1);

    assert(size % 2 == 0);

    double temp = 0;
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            temp = SIFT::gaussian2D(i - half_kernel_size, j - half_kernel_size, sigma);
            ret.at<float>(Point(j, i)) = (float)temp;
            sog += (float)temp;
        }
    }

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            float val = ret.at<float>(Point(j, i));
            ret.at<float>(Point(j, i)) = (float)(1.0 / (sog * val));
        }
    }

    return ret;
}

// gaussian2D
// Returns the value of the bell curve at a (x,y) for a given sigma
double SIFT::gaussian2D(double x, double y, double sigma)
{
    double ret = 1.0 / (2 * M_PI * sigma * sigma) * exp(-(x * x + y * y) / (2.0 * sigma * sigma));


    return ret;
}

// ExtractKeypointDescriptors()
// Generates a unique descriptor for each keypoint descriptor
/*
void meSIFT::ExtractKeypointDescriptors()
{
    vector<vector<Mat> >imginterpolatedMagnitude;
    vector<vector<Mat> >imginterpolatedOrientation;
    for (int i = 0; i < m_numOctaves; i++)
    {
        imginterpolatedMagnitude.push_back(vector<Mat>());
        imginterpolatedOrientation.push_back(vector<Mat>());
        for (int j = 1; j < m_numIntervals + 1; j++)
        {
            Mat temp = Mat::zeros(m_glist[i][j].size()*2,CV_32FC1);

            // Scale it up. This will give us "access" to in betweens
            pyrUp(m_glist[i][j], temp);

            int width = m_glist[i][j].cols;
            int height = m_glist[i][j].rows;
            Mat temp1 = Mat::zeros(Size(width+1,height+1), CV_32FC1);
            Mat temp2 = Mat::zeros(Size(width + 1, height + 1), CV_32FC1);
            imginterpolatedMagnitude[i].push_back(temp1);
            imginterpolatedOrientation[i].push_back(temp2);

            // Do the calculations
            for (int ii = 1; ii < height - 1; ii++)
            {
                for (int jj = 1; jj < width - 1; jj++)
                {
                    // "inbetween" change
                    float dx = (float)(m_glist[i][j].at<float>(Point(jj,ii+1))+m_glist[i][j].at<float>(Point(jj,ii)))/2.0-(float)(m_glist[i][j].at<float>(Point(jj,ii))+m_glist[i][j].at<float>(Point(jj,ii-1)))/2.0;
                    float dy = (float)(m_glist[i][j].at<float>(Point(jj,ii+1))+m_glist[i][j].at<float>(Point(jj,ii)))/2.0-(float)(m_glist[i][j].at<float>(Point(jj,ii))+m_glist[i][j].at<float>(Point(jj,ii-1)))/2.0;

                    int iii = ii + 1;
                    int jjj = jj + 1;
                    if (iii <= height && jjj <= width)
                    {
                        imginterpolatedMagnitude[i][j - 1].at<float>(Point(jjj, iii)) = sqrt(dx*dx+dy*dy);
                        imginterpolatedOrientation[i][j-1].at<float>(Point(jjj,iii))= (atan2(dy, dx) == M_PI) ? float(-M_PI) : float(atan2(dy, dx));

                    }
                }
            }

            // Pad the edges with zeros
            for (int iii = 0; iii < height+1; iii++)
            {
                imginterpolatedMagnitude[i][j-1].at<float>(Point(0,iii))= 0.0;
                imginterpolatedMagnitude[i][j-1].at<float>(Point(width,iii))= 0.0;
                imginterpolatedOrientation[i][j-1].at<float>(Point(0,iii))= 0.0;
                imginterpolatedOrientation[i][j-1].at<float>(Point(width,iii))= 0.0;

            }

            for (int jjj = 0; jjj < width+1; jjj++)
            {
                imginterpolatedMagnitude[i][j - 1].at<float>(Point(jjj, 0)) = 0.0;
                imginterpolatedMagnitude[i][j - 1].at<float>(Point(jjj,height)) = 0.0;
                imginterpolatedOrientation[i][j - 1].at<float>(Point(jjj,0)) = 0.0;
                imginterpolatedOrientation[i][j - 1].at<float>(Point(jjj,height)) = 0.0;
            }

            //imshow(to_string(10 * j + i),imginterpolatedMagnitude[i][j-1]);
            //waitKey(0);
            //imshow(to_string(10 * j + i),imginterpolatedOrientation[i][j-1]);
            //waitKey(0);
        }
    }

    // Build an Interpolated Gaussian Table of size FEATURE_WINDOW_SIZE
    // Lowe suggests sigma should be half the window size
    // This is used to construct the "circular gaussian window" to weight
    // magnitudes
    Mat G = meSIFT::BuildInterpolatedGaussianTable(FEATURE_WINDOW_SIZE, 0.5 * FEATURE_WINDOW_SIZE);
    vector<float> hist(DESC_NUM_BINS);
    for (int ikp = 0; ikp < m_numKeypoints; ikp++)
    {
        int scale = m_keyPoints[ikp].scale;
        float kpxi = m_keyPoints[ikp].xi;
        float kpyi = m_keyPoints[ikp].yi;

        float descxi = kpxi;
        float descyi = kpyi;

        int ii = (int)(kpxi * 2) / (int)(pow(2.0, (double)scale / m_numIntervals));
        int jj = (int)(kpyi * 2) / (int)(pow(2.0, (double)scale / m_numIntervals));

        int width = m_glist[scale / m_numIntervals][0].cols;
        int height = m_glist[scale / m_numIntervals][0].rows;

        vector<float> orien = m_keyPoints[ikp].orien;
        vector<float> magn = m_keyPoints[ikp].mag;

        //cout << orien[2] << endl;
        //cout << magn[0] << endl;
        // Find the orientation and magnitude that have the maximum impact
        // on the feature
        float main_magn = INT_MIN;
        float main_orien = INT_MIN;
        for (int orient_count = 1; orient_count < magn.size(); orient_count++)
        {
            if (magn[orient_count] > main_magn)
            {
                main_orien = orien[orient_count];
                main_magn = magn[orient_count];
            }
        }
        int hfsz = FEATURE_WINDOW_SIZE / 2;
        Mat weight = Mat(Size(FEATURE_WINDOW_SIZE, FEATURE_WINDOW_SIZE), CV_32FC1);
        vector<float> fv(FVSIZE);
        int i, j;
        for (i = 0; i < FEATURE_WINDOW_SIZE; i++)
        {
            for (j = 0; j < FEATURE_WINDOW_SIZE; j++)
            {
                if (ii + i + 1 < hfsz || ii + i + 1 > width + hfsz || jj + j + 1 < hfsz || jj + j + 1 > height + hfsz)
                    weight.at<float>(Point(j, i)) = 0;
                else
                {
                    float val1 = G.at<float>(Point(j, i));
                    float val2 = imginterpolatedMagnitude[scale / m_numIntervals][scale % m_numIntervals].at<float>(Point(jj + j + 1 - hfsz, ii + i + 1 - hfsz));
                    weight.at<float>(Point(j, i)) = val1 * val2;
                }
            }
        }

        // Now that we've weighted the required magnitudes, we proceed to generating
        // the feature vector

        // The next two two loops are for splitting the 16x16 window
        // into sixteen 4x4 blocks
        for (i = 0; i < (int)FEATURE_WINDOW_SIZE / 4; i++)			// 4x4 thingy
        {
            for (j = 0; j < (int)FEATURE_WINDOW_SIZE / 4; j++)
            {
                // Clear the histograms
                for (int t = 0; t < DESC_NUM_BINS; t++)
                    hist[t] = 0.0;

                // Calculate the coordinates of the 4x4 block
                int starti = (int)ii - (int)hfsz + 1 + (int)(hfsz / 2 * i);
                int startj = (int)jj - (int)hfsz + 1 + (int)(hfsz / 2 * j);
                int limiti = (int)ii + (int)(hfsz / 2) * ((int)(i)-1);
                int limitj = (int)jj + (int)(hfsz / 2) * ((int)(j)-1);

                // Go though this 4x4 block and do the thingy :D
                for (int k = starti; k <= limiti; k++)
                {
                    for (int t = startj; t <= limitj; t++)
                    {
                        if (k<0 || k>width || t<0 || t>height)
                            continue;

                        // This is where rotation invariance is done
                        float sample_orien = imginterpolatedOrientation[scale / m_numIntervals][scale % m_numIntervals].at<float>(Point(t,k));
                        sample_orien -= main_orien;

                        if (sample_orien < 0)
                        {
                            while (sample_orien < 0)
                                sample_orien +=(float) 2 * M_PI;
                        }

                            if (sample_orien > 2 * M_PI)
                            {
                                sample_orien = sample_orien - float(2 * M_PI);
                            }

                            int sample_orien_d=0;
                            int bin=0;
                            float bin_f=0.0;
                            float vxls=0.0;

                        // This should never happen
                        //if (!(sample_orien >= 0 && sample_orien < 2 * M_PI))
                            //printf("BAD: %f\n", sample_orien);
                            if (sample_orien >= 0 && sample_orien < 2 * M_PI)
                            {
                                sample_orien_d = sample_orien * 180 / M_PI;
                                if (sample_orien_d < 360)
                                {
                                    bin = sample_orien_d / (360 / DESC_NUM_BINS);
                                    bin_f= (float)sample_orien_d / (float)(360 / DESC_NUM_BINS);		// The actual entry// The bin
                                }
                            }

                            if (bin < DESC_NUM_BINS && k + hfsz - 1 - ii < FEATURE_WINDOW_SIZE && t + hfsz - 1 - jj < FEATURE_WINDOW_SIZE)
                            {
                                // Add to the bin
                                vxls = weight.at<float>(Point(t + hfsz - 1 - jj, k + hfsz - 1 - ii));
                                hist[bin] += (1 - fabs(bin_f - (bin + 0.5))) * vxls;
                            }
                    }
                }

                // Keep adding these numbers to the feature vector
                for (int t = 0; t < DESC_NUM_BINS; t++)
                {
                    fv[(i * FEATURE_WINDOW_SIZE / 4 + j) * DESC_NUM_BINS + t] = hist[t];
                }
            }
        }
        // Now, normalize the feature vector to ensure illumination independence
        float norm = 0;
        for (int t = 0; t < FVSIZE; t++)
            norm +=(float) pow(fv[t], 2.0);
        norm = sqrt(norm);

        for (int t = 0; t < FVSIZE; t++)
            fv[t] /= norm;

        // Now, threshold the vector
        for (int t = 0; t < FVSIZE; t++)
            if (fv[t] > FV_THRESHOLD)
                fv[t] = FV_THRESHOLD;

        // Normalize yet again
        norm = 0;
        for (int t = 0; t < FVSIZE; t++)
            norm += pow(fv[t], 2.0);
        norm = sqrt(norm);

        for (int t = 0; t < FVSIZE; t++)
            fv[t] /= norm;

        // We're done with this descriptor. Store it into a list
        Descriptor k(descxi, descyi, fv);
        m_keyDescs.push_back(k);
    }

    assert(m_keyDescs.size() == m_numKeypoints);
}
*/
// ShowKeypoints()
// Displays keypoints as calculated by the algorithm

void SIFT::ShowKeypoints()
{
    Mat img = srcImage.clone();

    for (int i = 0; i < m_keyPoints.size(); i++)
    {
        Keypoint kp = m_keyPoints[i];
        circle(img, Point(kp.xi, kp.yi),6,Scalar(0,255,0),1);
        //line(img, Point(kp.xi, kp.yi), Point(kp.xi, kp.yi), CV_RGB(255, 255, 255), 3);
        if (kp.orien.empty()) continue;
        else
        {
            line(img, Point(kp.xi, kp.yi), Point(kp.xi+10 *cos(kp.orien[0]), kp.yi + 10 * sin((float)kp.orien[0])), CV_RGB(255, 255, 255), 1);
        }
    }

    imshow("Keypoints", img);
    waitKey(0);
}



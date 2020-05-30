#pragma once
#ifndef _KEYPOINT_H
#define _KEYPOINT_H

#include "stdafx.h"
#include <vector>

using namespace std;

class Keypoint
{
public:
    float			xi;
    float			yi;	// It's location
    vector<float>	mag;	// The list of magnitudes at this point
    vector<float>	orien;	// The list of orientations detected
    int	scale;	// The scale where this was detected

    //defining the constructor
    Keypoint(float x, float y, vector<float>m, vector<float>o, int s)
    {
        xi = x;
        yi = y;
        mag = m;
        orien = o;
        scale = s;
    }
};

#endif

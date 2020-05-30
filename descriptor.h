#pragma once
#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H
#include<vector>
using namespace std;

class Descriptor
{
public:
    float			xi, yi;		// The location
    vector<float>	fv;			// The feature vector

    Descriptor(float x, float y, vector<float> const& f)
    {
        xi = x;
        yi = y;
        fv = f;
    }
};

#endif

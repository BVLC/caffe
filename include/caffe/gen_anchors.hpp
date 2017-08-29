#ifndef GEN_ANCHORS
#define GEN_ANCHORS

#include <vector>

using namespace std;

namespace caffe {

/**
 * @brief Type of faster-rcnn anchor
 */
struct anchor {
    float start_x;
    float start_y;
    float end_x;
    float end_y;

    anchor() {}

    anchor(float s_x, float s_y, float e_x, float e_y)
    {
        start_x = s_x;
        start_y = s_y;
        end_x   = e_x;
        end_y   = e_y;
    }
};


/**
 * @brief Generates a vector of anchors based on a size, list of ratios and list of scales
 */
void GenerateAnchors(unsigned int base_size, const vector<float>& ratios, const vector<float> scales,   // input
                     anchor *anchors);                                                                  // output
}

#endif

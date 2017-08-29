#include <math.h>
#include <vector>

#include "caffe/gen_anchors.hpp"

namespace caffe {


static void CalcBasicParams(const anchor& base_anchor,                                       // input
                            float& width, float& height, float& x_center, float& y_center)   // output
{
    width  = base_anchor.end_x - base_anchor.start_x + 1.0f;
    height = base_anchor.end_y - base_anchor.start_y + 1.0f;

    x_center = base_anchor.start_x + 0.5f * (width - 1.0f);
    y_center = base_anchor.start_y + 0.5f * (height - 1.0f);
}


static void MakeAnchors(const vector<float>& ws, const vector<float>& hs, float x_center, float y_center,   // input
                        vector<anchor>& anchors)                                                            // output
{
    int len = ws.size();
    anchors.clear();
    anchors.resize(len);

    for (unsigned int i = 0 ; i < len ; i++) {
        // transpose to create the anchor
        anchors[i].start_x = x_center - 0.5f * (ws[i] - 1.0f);
        anchors[i].start_y = y_center - 0.5f * (hs[i] - 1.0f);
        anchors[i].end_x   = x_center + 0.5f * (ws[i] - 1.0f);
        anchors[i].end_y   = y_center + 0.5f * (hs[i] - 1.0f);
    }
}


static void CalcAnchors(const anchor& base_anchor, const vector<float>& scales,        // input
                        vector<anchor>& anchors)                                       // output
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    CalcBasicParams(base_anchor, width, height, x_center, y_center);

    int num_scales = scales.size();
    vector<float> ws(num_scales), hs(num_scales);

    for (unsigned int i = 0 ; i < num_scales ; i++) {
        ws[i] = width * scales[i];
        hs[i] = height * scales[i];
    }

    MakeAnchors(ws, hs, x_center, y_center, anchors);
}


static void CalcRatioAnchors(const anchor& base_anchor, const vector<float>& ratios,        // input
                             vector<anchor>& ratio_anchors)                                 // output
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    CalcBasicParams(base_anchor, width, height, x_center, y_center);

    float size = width * height;

    int num_ratios = ratios.size();

    vector<float> ws(num_ratios), hs(num_ratios);

    for (unsigned int i = 0 ; i < num_ratios ; i++) {
        float new_size = size / ratios[i];
        ws[i] = round(sqrt(new_size));
        hs[i] = round(ws[i] * ratios[i]);
    }

    MakeAnchors(ws, hs, x_center, y_center, ratio_anchors);
}

void GenerateAnchors(unsigned int base_size, const vector<float>& ratios, const vector<float> scales,   // input
                     anchor *anchors)                                                           // output
{
    float end = (float)(base_size - 1);        // because we start at zero

    anchor base_anchor(0.0f, 0.0f, end, end);

    vector<anchor> ratio_anchors;
    CalcRatioAnchors(base_anchor, ratios, ratio_anchors);

    for (int i = 0, index = 0; i < ratio_anchors.size() ; i++) {
        vector<anchor> temp_anchors;
        CalcAnchors(ratio_anchors[i], scales, temp_anchors);

        for (int j = 0 ; j < temp_anchors.size() ; j++) {
            anchors[index++] = temp_anchors[j];
        }
    }
}

}  // namespace caffe

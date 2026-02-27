// DeformConv (deformable convolution 2D) compute shader.
// One thread per output element [N, M, out_h, out_w].
// Reference: ONNX DeformConv spec, CPU/CUDA implementations (im2col + bilinear + GEMM).
#if !defined(TBUFFER)
#define TBUFFER float
#endif

RWStructuredBuffer<TBUFFER> input_x   : register(u0);  // [N, C, H, W]
RWStructuredBuffer<TBUFFER> weight   : register(u1);   // [M, C/group, kH, kW]
RWStructuredBuffer<TBUFFER> offset   : register(u2);  // [N, offset_groups*2*kH*kW, out_h, out_w]
RWStructuredBuffer<TBUFFER> mask     : register(u3);  // [N, offset_groups*kH*kW, out_h, out_w] (optional)
RWStructuredBuffer<TBUFFER> bias     : register(u4);  // [M] (optional)
RWStructuredBuffer<TBUFFER> output_y: register(u5);   // [N, M, out_h, out_w]

// Single uint4 array to tightly pack 40 uints and match C++ array of 40 uints
cbuffer Constants
{
    uint4 Constants[10];
};
#define N          Constants[0].x
#define C          Constants[0].y
#define M          Constants[0].z
#define H          Constants[0].w
#define W          Constants[1].x
#define kH         Constants[1].y
#define kW         Constants[1].z
#define out_h      Constants[1].w
#define out_w      Constants[2].x
#define pad_h      ((int)Constants[2].y)
#define pad_w      ((int)Constants[2].z)
#define stride_h   Constants[2].w
#define stride_w   Constants[3].x
#define dilation_h Constants[3].y
#define dilation_w Constants[3].z
#define group      Constants[3].w
#define offset_group Constants[4].x
#define use_mask   Constants[4].y
#define has_bias   Constants[4].z
#define InputStrides   uint4(Constants[4].w, Constants[5].x, Constants[5].y, Constants[5].z)
#define WeightStrides  uint4(Constants[5].w, Constants[6].x, Constants[6].y, Constants[6].z)
#define OffsetStrides  uint4(Constants[6].w, Constants[7].x, Constants[7].y, Constants[7].z)
#define MaskStrides    uint4(Constants[7].w, Constants[8].x, Constants[8].y, Constants[8].z)
#define OutputStrides  uint4(Constants[8].w, Constants[9].x, Constants[9].y, Constants[9].z)
#define start_idx      Constants[9].w

// Bilinear interpolation at (h, w). Returns 0 if out of bounds (ONNX spec).
float SampleInput(uint n, uint c, float h, float w)
{
    if (h <= -1.0 || h >= (float)H || w <= -1.0 || w >= (float)W)
        return 0.0;

    int h_low = (int)floor(h);
    int w_low = (int)floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - (float)h_low;
    float lw = w - (float)w_low;
    float hh = 1.0 - lh;
    float hw = 1.0 - lw;

    float v1 = (h_low >= 0 && w_low >= 0) ? (float)input_x[n * InputStrides.x + c * InputStrides.y + (uint)h_low * InputStrides.z + (uint)w_low] : 0.0;
    float v2 = (h_low >= 0 && w_high < (int)W) ? (float)input_x[n * InputStrides.x + c * InputStrides.y + (uint)h_low * InputStrides.z + (uint)w_high] : 0.0;
    float v3 = (h_high < (int)H && w_low >= 0) ? (float)input_x[n * InputStrides.x + c * InputStrides.y + (uint)h_high * InputStrides.z + (uint)w_low] : 0.0;
    float v4 = (h_high < (int)H && w_high < (int)W) ? (float)input_x[n * InputStrides.x + c * InputStrides.y + (uint)h_high * InputStrides.z + (uint)w_high] : 0.0;

    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

[numthreads(64, 1, 1)]
void DeformConv(uint3 dtid : SV_DispatchThreadId)
{
    uint out_idx = start_idx + dtid.x;
    uint totalOut = N * M * out_h * out_w;
    if (out_idx >= totalOut)
        return;
    
    // Prevent divide by zero errors on empty tensors
    if (out_w == 0 || out_h == 0 || M == 0 || N == 0)
        return;

    uint tmp = out_idx;
    uint ow = tmp % max(out_w, 1u);
    tmp /= max(out_w, 1u);
    uint oh = tmp % max(out_h, 1u);
    tmp /= max(out_h, 1u);
    uint m = tmp % max(M, 1u);
    uint n = tmp / max(M, 1u);

    uint groups = max(group, 1u);
    uint outChPerGroup = max(M / groups, 1u);
    uint inChPerGroup = max(C / groups, 1u);
    uint g = outChPerGroup > 0 ? m / outChPerGroup : 0;
    uint mLocal = outChPerGroup > 0 ? m % outChPerGroup : 0;

    uint channelPerOffsetGroup = max(C / max(offset_group, 1u), 1u);
    uint kernelSize = kH * kW;

    float acc = 0.0;

    for (uint cLocal = 0; cLocal < inChPerGroup; cLocal++)
    {
        uint c = g * inChPerGroup + cLocal;
        uint offsetGrp = c / channelPerOffsetGroup;

        for (uint i = 0; i < kH; i++)
        {
            for (uint j = 0; j < kW; j++)
            {
                uint kj = i * kW + j;

                // Offset layout: [N, offset_groups*2*kH*kW, out_h, out_w], h and w interleaved per kernel pos
                uint offsetBase = n * OffsetStrides.x
                    + (offsetGrp * 2 * kernelSize + 2 * kj) * OffsetStrides.y
                    + oh * OffsetStrides.z + ow;
                float off_h = (float)offset[offsetBase];
                float off_w = (float)offset[offsetBase + OffsetStrides.y];

                float maskVal = 1.0;
                if (use_mask != 0)
                {
                    uint maskIdx = n * MaskStrides.x
                        + (offsetGrp * kernelSize + kj) * MaskStrides.y
                        + oh * MaskStrides.z + ow;
                    maskVal = (float)mask[maskIdx];
                }

                float h_im = (float)((int)oh * (int)stride_h - pad_h) + (float)(i * dilation_h) + off_h;
                float w_im = (float)((int)ow * (int)stride_w - pad_w) + (float)(j * dilation_w) + off_w;

                float sampled = SampleInput(n, c, h_im, w_im);

                // Weight [M, C/group, kH, kW]
                uint wIdx = m * WeightStrides.x + cLocal * WeightStrides.y + i * WeightStrides.z + j;
                float wVal = (float)weight[wIdx];

                acc += wVal * sampled * maskVal;
            }
        }
    }

    if (has_bias != 0)
        acc += (float)bias[m];

    output_y[out_idx] = (TBUFFER)acc;
}

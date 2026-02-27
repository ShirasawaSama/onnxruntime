#pragma once

#include "../../../OperatorAuthorHelper/OperatorHelper.h"
#include "../MLOperatorAuthorImpl.h"
#include "../DmlCommon.h"
#include "../External/D3DX12/d3dx12.h"
#include "directx/d3d12.h"

// Shader headers produced by GeneratedShaders/GenerateShaders.bat
namespace DeformConv_float
{
    #include "GeneratedShaders/deform_conv_float.h"
}
namespace DeformConv_fp16
{
    #include "GeneratedShaders/deform_conv_fp16.h"
}

#include <wrl/client.h>
#include <wrl/implements.h>
using namespace Microsoft::WRL;

enum DmlDeformConvInputIndex : uint32_t
{
    X = 0,
    W = 1,
    Offset = 2,
    B = 3,
    Mask = 4,
};

struct DmlDeformConvParameters
{
    uint32_t N = 0, C = 0, M = 0, H = 0, W = 0;
    uint32_t kH = 0, kW = 0, out_h = 0, out_w = 0;
    int32_t pad_h = 0, pad_w = 0;
    uint32_t stride_h = 1, stride_w = 1;
    uint32_t dilation_h = 1, dilation_w = 1;
    uint32_t group = 1, offset_group = 1;

    DmlDeformConvParameters() = default;

    DmlDeformConvParameters(
        const OperatorHelper::IKernelInformationAdapter& kernelInfo,
        const OperatorHelper::IShapeInformationAdapter& shapeInfo)
    {
        auto& attributes = kernelInfo.GetAttributes();
        std::vector<int> kernelShape = attributes.GetOptionalAttributeVectorInt32(AttrName::KernelShape);
        std::vector<int> strides = attributes.GetOptionalAttributeVectorInt32(AttrName::Strides);
        std::vector<int> pads = attributes.GetOptionalAttributeVectorInt32(AttrName::Pads);
        std::vector<int> dilations = attributes.GetOptionalAttributeVectorInt32(AttrName::Dilations);

        group = gsl::narrow_cast<uint32_t>(attributes.GetOptionalAttribute<int64_t>(AttrName::Group, 1));
        offset_group = gsl::narrow_cast<uint32_t>(attributes.GetOptionalAttribute<int64_t>("offset_group", 1));

        auto xDims = shapeInfo.GetInputTensorShape(DmlDeformConvInputIndex::X);
        auto wDims = shapeInfo.GetInputTensorShape(DmlDeformConvInputIndex::W);
        (void)shapeInfo.GetInputTensorShape(DmlDeformConvInputIndex::Offset);

        N = gsl::narrow_cast<uint32_t>(xDims[0]);
        C = gsl::narrow_cast<uint32_t>(xDims[1]);
        H = gsl::narrow_cast<uint32_t>(xDims[2]);
        W = gsl::narrow_cast<uint32_t>(xDims[3]);
        M = gsl::narrow_cast<uint32_t>(wDims[0]);
        kH = kernelShape.size() >= 1 ? gsl::narrow_cast<uint32_t>(kernelShape[0]) : gsl::narrow_cast<uint32_t>(wDims[2]);
        kW = kernelShape.size() >= 2 ? gsl::narrow_cast<uint32_t>(kernelShape[1]) : gsl::narrow_cast<uint32_t>(wDims[3]);

        stride_h = strides.size() >= 1 ? std::max(1u, gsl::narrow_cast<uint32_t>(strides[0])) : 1;
        stride_w = strides.size() >= 2 ? std::max(1u, gsl::narrow_cast<uint32_t>(strides[1])) : 1;
        pad_h = pads.size() >= 1 ? pads[0] : 0;
        pad_w = pads.size() >= 2 ? pads[1] : 0;
        int pad_h_end = pads.size() >= 4 ? pads[2] : 0;
        int pad_w_end = pads.size() >= 4 ? pads[3] : 0;
        dilation_h = dilations.size() >= 1 ? gsl::narrow_cast<uint32_t>(dilations[0]) : 1;
        dilation_w = dilations.size() >= 2 ? gsl::narrow_cast<uint32_t>(dilations[1]) : 1;

        if (stride_h > 0 && stride_w > 0) {
            int out_h_int = (int)(H + pad_h + pad_h_end - (int)(dilation_h * (kH - 1)) - 1) / (int)stride_h + 1;
            int out_w_int = (int)(W + pad_w + pad_w_end - (int)(dilation_w * (kW - 1)) - 1) / (int)stride_w + 1;
            out_h = gsl::narrow_cast<uint32_t>(std::max(0, out_h_int));
            out_w = gsl::narrow_cast<uint32_t>(std::max(0, out_w_int));
        } else {
            out_h = 0;
            out_w = 0;
        }
    }
};

namespace DeformConvHelpers
{
    inline uint32_t CeilDivide(uint32_t dividend, uint32_t divisor)
    {
        return (dividend + divisor - 1) / divisor;
    }

    inline void GetNextDispatchSize(
        uint32_t elementCount,
        uint32_t elementsPerThread,
        uint32_t numThreads,
        _Out_ uint32_t& dispatch,
        _Out_ uint32_t& pendingElementCount)
    {
        const uint32_t maxThreadsPerDispatch = numThreads * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
        const uint32_t requiredThreadCount = CeilDivide(elementCount, elementsPerThread);
        const uint32_t availableThreadCount = std::min(requiredThreadCount, maxThreadsPerDispatch);
        uint32_t workGroupCount1D = CeilDivide(availableThreadCount, numThreads);
        dispatch = workGroupCount1D;
        const uint32_t dispatchedElementCount = workGroupCount1D * numThreads * elementsPerThread;
        pendingElementCount = (dispatchedElementCount < elementCount) ? elementCount - dispatchedElementCount : 0;
    }
}

// Layout must match HLSL Constants[40] exactly (avoids cbuffer packing mismatch).
struct DeformConvShaderConstants
{
    uint32_t data[40];
    void set(uint32_t N_, uint32_t C_, uint32_t M_, uint32_t H_, uint32_t W_,
             uint32_t kH_, uint32_t kW_, uint32_t out_h_, uint32_t out_w_,
             int32_t pad_h_, int32_t pad_w_, uint32_t stride_h_, uint32_t stride_w_,
             uint32_t dilation_h_, uint32_t dilation_w_, uint32_t group_, uint32_t offset_group_,
             uint32_t use_mask_, uint32_t has_bias_,
             const uint32_t* inputStrides, const uint32_t* weightStrides,
             const uint32_t* offsetStrides, const uint32_t* maskStrides, const uint32_t* outputStrides, uint32_t start_idx)
    {
        data[0] = N_; data[1] = C_; data[2] = M_; data[3] = H_; data[4] = W_;
        data[5] = kH_; data[6] = kW_; data[7] = out_h_; data[8] = out_w_;
        data[9] = (uint32_t)pad_h_; data[10] = (uint32_t)pad_w_;
        data[11] = stride_h_; data[12] = stride_w_;
        data[13] = dilation_h_; data[14] = dilation_w_;
        data[15] = group_; data[16] = offset_group_;
        data[17] = use_mask_; data[18] = has_bias_;
        for (int i = 0; i < 4; i++) {
            data[19 + i] = inputStrides[i];
            data[23 + i] = weightStrides[i];
            data[27 + i] = offsetStrides[i];
            data[31 + i] = maskStrides[i];
            data[35 + i] = outputStrides[i];
        }
        data[39] = start_idx;
    }
};

class DmlDeformConvOperator : public WRL::Base<IMLOperatorKernel>
{
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pso;
    ComPtr<ID3D12Resource> m_dummyBuffer;
    DmlDeformConvParameters m_params;

public:
    DmlDeformConvOperator(IMLOperatorKernelCreationContext* context)
    {
        ComPtr<IUnknown> executionObject;
        context->GetExecutionInterface(executionObject.GetAddressOf());
        ComPtr<ID3D12GraphicsCommandList> commandList;
        executionObject.As(&commandList);
        ORT_THROW_IF_FAILED(commandList->GetDevice(IID_ID3D12Device, &m_device));

        MLOperatorKernelCreationContext creationContext(context);
        OperatorHelper::KernelInformationAdapter kernelInfo{creationContext};
        OperatorHelper::ShapeInformationAdapter shapeInfo{creationContext};
        m_params = DmlDeformConvParameters(kernelInfo, shapeInfo);

        MLOperatorEdgeDescription edgeDesc;
        ORT_THROW_IF_FAILED(context->GetInputEdgeDescription(0, &edgeDesc));
        PrepareDeformConv(edgeDesc.tensorDataType);
    }

    void PrepareDeformConv(MLOperatorTensorDataType dataType)
    {
        const uint32_t uavCount = 6;
        std::vector<CD3DX12_ROOT_PARAMETER1> rootParameters(uavCount + 1);
        for (uint32_t i = 0; i < uavCount; i++)
            rootParameters[i].InitAsUnorderedAccessView(i);
        const int constantCount = 40;
        rootParameters[uavCount].InitAsConstants(constantCount, 0);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        desc.Init_1_1(static_cast<uint32_t>(rootParameters.size()), rootParameters.data());
        ComPtr<ID3DBlob> rootSignatureBlob, rootSignatureErrorBlob;
        ORT_THROW_IF_FAILED(D3D12SerializeVersionedRootSignature(&desc, &rootSignatureBlob, &rootSignatureErrorBlob));
        ORT_THROW_IF_FAILED(m_device->CreateRootSignature(0, rootSignatureBlob->GetBufferPointer(), rootSignatureBlob->GetBufferSize(), IID_ID3D12RootSignature, &m_rootSignature));

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = m_rootSignature.Get();
        if (dataType == MLOperatorTensorDataType::Float16)
            psoDesc.CS = CD3DX12_SHADER_BYTECODE(DeformConv_fp16::g_DeformConv, sizeof(DeformConv_fp16::g_DeformConv));
        else
            psoDesc.CS = CD3DX12_SHADER_BYTECODE(DeformConv_float::g_DeformConv, sizeof(DeformConv_float::g_DeformConv));
        ORT_THROW_IF_FAILED(m_device->CreateComputePipelineState(&psoDesc, IID_ID3D12PipelineState, &m_pso));

        D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_RESOURCE_DESC resDesc = CD3DX12_RESOURCE_DESC::Buffer(16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ORT_THROW_IF_FAILED(m_device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_ID3D12Resource, &m_dummyBuffer));
    }

    STDMETHOD(Compute)(IMLOperatorKernelContext* context) override
    {
        try
        {
            ComPtr<IMLOperatorTensor> inputX, weight, offset, bias, mask, outputY;
            ORT_THROW_IF_FAILED(context->GetInputTensor(0, &inputX));
            ORT_THROW_IF_FAILED(context->GetInputTensor(1, &weight));
            ORT_THROW_IF_FAILED(context->GetInputTensor(2, &offset));
            context->GetInputTensor(3, &bias);
            context->GetInputTensor(4, &mask);
            context->GetOutputTensor(0, &outputY);

            if (inputX->IsCpuData() || weight->IsCpuData() || offset->IsCpuData())
                return E_UNEXPECTED;

            ComPtr<IUnknown> executionObject;
            ComPtr<ID3D12GraphicsCommandList> commandList;
            context->GetExecutionInterface(executionObject.GetAddressOf());
            executionObject.As(&commandList);

            auto xDims = GetTensorDimensions(inputX.Get());
            auto wDims = GetTensorDimensions(weight.Get());
            auto offsetDims = GetTensorDimensions(offset.Get());
            auto outDims = GetTensorDimensions(outputY.Get());
            std::vector<uint32_t> maskDims;
            if (mask) maskDims = GetTensorDimensions(mask.Get());

            ComPtr<ID3D12Resource> resX, resW, resOffset, resMask, resBias, resOut;
            GetResourceFromTensor(inputX.Get(), &resX);
            GetResourceFromTensor(weight.Get(), &resW);
            GetResourceFromTensor(offset.Get(), &resOffset);
            GetResourceFromTensor(outputY.Get(), &resOut);
            if (bias) GetResourceFromTensor(bias.Get(), &resBias);
            if (mask) GetResourceFromTensor(mask.Get(), &resMask);

            DeformConv(resX.Get(), resW.Get(), resOffset.Get(),
                resBias.Get(), resMask.Get(), resOut.Get(),
                xDims, wDims, offsetDims, maskDims, outDims, commandList.Get());
        }
        catch (...) { return E_FAIL; }
        return S_OK;
    }

private:
    std::vector<uint32_t> GetTensorDimensions(IMLOperatorTensor* tensor)
    {
        uint32_t dimCount = tensor->GetDimensionCount();
        std::vector<uint32_t> dims(dimCount);
        ORT_THROW_IF_FAILED(tensor->GetShape(dimCount, dims.data()));
        return dims;
    }

    void GetResourceFromTensor(IMLOperatorTensor* tensor, ID3D12Resource** ppResource)
    {
        ComPtr<IUnknown> unknown;
        tensor->GetDataInterface(unknown.GetAddressOf());
        unknown.CopyTo(ppResource);
    }

    void DeformConv(
        ID3D12Resource* resX, ID3D12Resource* resW, ID3D12Resource* resOffset,
        ID3D12Resource* resBias, ID3D12Resource* resMask, ID3D12Resource* resOut,
        const std::vector<uint32_t>& xDims, const std::vector<uint32_t>& wDims,
        const std::vector<uint32_t>& offsetDims, const std::vector<uint32_t>& maskDims, const std::vector<uint32_t>& outDims,
        ID3D12GraphicsCommandList* commandList)
    {
        std::array<uint32_t, 4> inputStrides, weightStrides, offsetStrides, maskStrides, outputStrides;
        Dml::GetDescendingPackedStrides(gsl::span<const uint32_t>(xDims), gsl::span<uint32_t>(inputStrides));
        Dml::GetDescendingPackedStrides(gsl::span<const uint32_t>(wDims), gsl::span<uint32_t>(weightStrides));
        Dml::GetDescendingPackedStrides(gsl::span<const uint32_t>(offsetDims), gsl::span<uint32_t>(offsetStrides));
        Dml::GetDescendingPackedStrides(gsl::span<const uint32_t>(outDims), gsl::span<uint32_t>(outputStrides));
        if (!maskDims.empty())
            Dml::GetDescendingPackedStrides(gsl::span<const uint32_t>(maskDims), gsl::span<uint32_t>(maskStrides));
        else
            maskStrides = offsetStrides;

        ID3D12Resource* uavs[6] = { resX, resW, resOffset, resMask, resBias, resOut };

        uint32_t numBarriers = 0;
        D3D12_RESOURCE_BARRIER barriers[6];
        ID3D12Resource* uniqueUavs[6] = { nullptr };
        for (uint32_t i = 0; i < 6; i++)
        {
            if (uavs[i] == nullptr) continue;
            
            bool duplicate = false;
            for (uint32_t j = 0; j < numBarriers; j++)
            {
                if (uniqueUavs[j] == uavs[i])
                {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate)
            {
                uniqueUavs[numBarriers] = uavs[i];
                barriers[numBarriers] = CD3DX12_RESOURCE_BARRIER::Transition(uavs[i], D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                numBarriers++;
            }
        }
        
        if (numBarriers > 0)
            commandList->ResourceBarrier(numBarriers, barriers);

        commandList->SetComputeRootSignature(m_rootSignature.Get());
        commandList->SetPipelineState(m_pso.Get());

        for (uint32_t i = 0; i < 6; i++)
        {
            if (uavs[i] != nullptr)
                commandList->SetComputeRootUnorderedAccessView(i, uavs[i]->GetGPUVirtualAddress());
            else
                commandList->SetComputeRootUnorderedAccessView(i, m_dummyBuffer->GetGPUVirtualAddress());
        }

        uint32_t elementCount = m_params.N * m_params.M * m_params.out_h * m_params.out_w;

        if (elementCount == 0 || m_params.out_w == 0 || m_params.out_h == 0 || m_params.M == 0 || m_params.N == 0) return;

        DeformConvShaderConstants constants;
        constants.set(m_params.N, m_params.C, m_params.M, m_params.H, m_params.W,
            m_params.kH, m_params.kW, m_params.out_h, m_params.out_w,
            m_params.pad_h, m_params.pad_w, m_params.stride_h, m_params.stride_w,
            m_params.dilation_h, m_params.dilation_w, m_params.group, m_params.offset_group,
            (resMask != nullptr) ? 1u : 0u, (resBias != nullptr) ? 1u : 0u,
            inputStrides.data(), weightStrides.data(), offsetStrides.data(),
            maskStrides.data(), outputStrides.data(), 0);

        uint32_t pendingCount = elementCount;
        while (pendingCount > 0)
        {
            uint32_t dispatchSizeX;
            uint32_t newPendingCount;
            DeformConvHelpers::GetNextDispatchSize(
                pendingCount,
                1,
                64,
                dispatchSizeX,
                newPendingCount
            );

            constants.data[39] = elementCount - pendingCount; // Update startIndex

            commandList->SetComputeRoot32BitConstants(6, 40, constants.data, 0);
            commandList->Dispatch(dispatchSizeX, 1, 1);

            pendingCount = newPendingCount;
        }

        for (uint32_t i = 0; i < numBarriers; i++)
            barriers[i] = CD3DX12_RESOURCE_BARRIER::Transition(uniqueUavs[i], D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        
        if (numBarriers > 0)
            commandList->ResourceBarrier(numBarriers, barriers);
    }
};

struct DeformConvShapeInferrer : public WRL::Base<IMLOperatorShapeInferrer>
{
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept override
    {
        try
        {
            MLShapeInferenceContext inferenceContext(context);
            OperatorHelper::KernelInformationAdapter kernelInfo{inferenceContext};
            OperatorHelper::ShapeInformationAdapter shapeInfo{inferenceContext};
            DmlDeformConvParameters params(kernelInfo, shapeInfo);
            std::array<uint32_t, 4> outputDims = { params.N, params.M, params.out_h, params.out_w };
            ORT_THROW_IF_FAILED(context->SetOutputTensorShape(0, 4, outputDims.data()));
        }
        catch (...) { return E_FAIL; }
        return S_OK;
    }
};

class DmlDeformConvOperatorFactory : public WRL::Base<IMLOperatorKernelFactory>
{
public:
    STDMETHOD(CreateKernel)(IMLOperatorKernelCreationContext* context, IMLOperatorKernel** kernel) override
    {
        try
        {
            auto op = wil::MakeOrThrow<DmlDeformConvOperator>(context);
            op.CopyTo(kernel);
            return S_OK;
        }
        catch (...) { return E_FAIL; }
    }

    static void RegisterDeformConvKernel(IMLOperatorRegistry* registry)
    {
        MLOperatorKernelDescription desc = {};
        desc.domain = "";
        desc.name = "DeformConv";
        desc.minimumOperatorSetVersion = 19;
        desc.executionType = MLOperatorExecutionType::D3D12;
        desc.options = MLOperatorKernelOptions::None;

        MLOperatorEdgeTypeConstrant tConstraint;
        tConstraint.typeLabel = "T";
        std::vector<MLOperatorEdgeDescription> tEdges = {
            { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float },
            { MLOperatorEdgeType::Tensor, (uint64_t)MLOperatorTensorDataType::Float16 },
        };
        tConstraint.allowedTypes = tEdges.data();
        tConstraint.allowedTypeCount = static_cast<uint32_t>(tEdges.size());
        std::vector<MLOperatorEdgeTypeConstrant> typeConstraints = { tConstraint };
        desc.typeConstraints = typeConstraints.data();
        desc.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

        auto shapeInferrer = wil::MakeOrThrow<DeformConvShapeInferrer>();
        auto factory = wil::MakeOrThrow<DmlDeformConvOperatorFactory>();
        ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
        ORT_THROW_IF_FAILED(registry->QueryInterface(registryPrivate.GetAddressOf()));
        ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(&desc, factory.Get(), shapeInferrer.Get(), nullptr, false, false));
    }
};

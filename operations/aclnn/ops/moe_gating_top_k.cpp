/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
==============================================================================*/

#include "moe_gating_top_k.h"

#include "acl/acl.h"
#include "aclnn_moe_gating_top_k.h"
#include "atb_speed/log.h"
#include "operations/aclnn/utils/utils.h"

namespace atb_speed {
namespace common {

MoeGatingTopKOperation::MoeGatingTopKOperation(
    const std::string &name, AclNNMoeGatingTopKParam param)
    : AclNNOperation(name), param_(param) {}

MoeGatingTopKOperation::~MoeGatingTopKOperation()
{
    ATB_SPEED_LOG_DEBUG("MoeGatingTopKOperation deconstructor");
    this->DestroyOperation();
}

uint32_t MoeGatingTopKOperation::GetInputNum() const
{
    return DIM2;
}

uint32_t MoeGatingTopKOperation::GetOutputNum() const
{
    return DIM3;
}

atb::Status MoeGatingTopKOperation::InferShape(
    const atb::SVector<atb::TensorDesc> &inTensorDescs,
    atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    outTensorDescs.at(DIM0).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(DIM0).dtype = aclDataType::ACL_FLOAT;
    outTensorDescs.at(DIM0).shape.dimNum = DIM2;
    outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM0).shape.dims[DIM1] = param_.k;

    outTensorDescs.at(DIM1).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(DIM1).dtype = aclDataType::ACL_INT32;
    outTensorDescs.at(DIM1).shape.dimNum = DIM2;
    outTensorDescs.at(DIM1).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM1).shape.dims[DIM1] = param_.k;

    outTensorDescs.at(DIM2).format = aclFormat::ACL_FORMAT_ND;
    outTensorDescs.at(DIM2).dtype = aclDataType::ACL_FLOAT;
    outTensorDescs.at(DIM2).shape.dimNum = DIM2;
    outTensorDescs.at(DIM2).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
    outTensorDescs.at(DIM2).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
    return atb::NO_ERROR;
}

atb::Status MoeGatingTopKOperation::CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack)
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    aclnnVariantPack.aclInTensors.resize(GetInputNum());
    for (size_t i = 0; i < aclnnVariantPack.aclInTensors.size(); ++i) {
        std::shared_ptr<AclNNTensor> aclnnTensor = std::make_shared<AclNNTensor>();
        aclnnTensor->tensorIdx = static_cast<int>(i);
        aclnnTensor->needUpdateTensorDataPtr = true;
        aclnnTensor->atbTensor = variantPack.inTensors.at(i);
        // aclnnMoeGatingTopK tiling requires bias to be 1D [expert_num].
        if (i == DIM1 && aclnnTensor->atbTensor.desc.shape.dimNum == DIM2 &&
            aclnnTensor->atbTensor.desc.shape.dims[DIM0] == 1) {
            aclnnTensor->atbTensor.desc.shape.dimNum = DIM1;
            aclnnTensor->atbTensor.desc.shape.dims[DIM0] =
                variantPack.inTensors.at(i).desc.shape.dims[DIM1];
        }
        aclnnTensor->strides = GetCopyTensorStride(aclnnTensor->atbTensor.desc.shape);
        aclnnTensor->tensor = aclCreateTensor(
            aclnnTensor->atbTensor.desc.shape.dims, aclnnTensor->atbTensor.desc.shape.dimNum,
            aclnnTensor->atbTensor.desc.dtype, aclnnTensor->strides.data(), 0,
            aclnnTensor->atbTensor.desc.format, aclnnTensor->atbTensor.desc.shape.dims,
            aclnnTensor->atbTensor.desc.shape.dimNum, aclnnTensor->atbTensor.deviceData);
        if (aclnnTensor->tensor == nullptr) {
            ATB_SPEED_LOG_ERROR(this->opName_ << " InTensor aclCreateTensor index " << i << " fail");
            return atb::ERROR_INTERNAL_ERROR;
        }
        aclnnVariantPack.aclInTensors[i] = aclnnTensor;
    }
    return atb::NO_ERROR;
}

int MoeGatingTopKOperation::SetAclNNWorkspaceExecutor()
{
    AclNNVariantPack &aclnnVariantPack = this->aclnnOpCache_->aclnnVariantPack;
    return aclnnMoeGatingTopKGetWorkspaceSize(
        aclnnVariantPack.aclInTensors[DIM0]->tensor,
        aclnnVariantPack.aclInTensors[DIM1]->tensor,
        param_.k,
        param_.kGroup,
        param_.groupCount,
        param_.groupSelectMode,
        param_.renorm,
        param_.normType,
        param_.outFlag,
        param_.routedScalingFactor,
        param_.eps,
        aclnnVariantPack.aclOutTensors.at(DIM0)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM1)->tensor,
        aclnnVariantPack.aclOutTensors.at(DIM2)->tensor,
        &this->aclnnOpCache_->workspaceSize,
        &this->aclnnOpCache_->aclExecutor);
}

int MoeGatingTopKOperation::ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream)
{
    return aclnnMoeGatingTopK(
        workspace, this->aclnnOpCache_->workspaceSize, this->aclnnOpCache_->aclExecutor, stream);
}

}  // namespace common
}  // namespace atb_speed

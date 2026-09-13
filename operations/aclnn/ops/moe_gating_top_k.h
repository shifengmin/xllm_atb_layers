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

#ifndef ATB_SPEED_PLUGIN_ACLNN_MOE_GATING_TOP_K_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_MOE_GATING_TOP_K_OPERATION_H

#include "operations/aclnn/core/acl_nn_operation.h"

namespace atb_speed::common {

struct AclNNMoeGatingTopKParam {
    int64_t k = 0;
    int64_t kGroup = 1;
    int64_t groupCount = 1;
    int64_t groupSelectMode = 1;
    int64_t renorm = 1;
    int64_t normType = 1;
    bool outFlag = false;
    double routedScalingFactor = 1.0;
    double eps = 1e-20;
};

class MoeGatingTopKOperation : public AclNNOperation {
public:
    explicit MoeGatingTopKOperation(const std::string &name, AclNNMoeGatingTopKParam param);
    ~MoeGatingTopKOperation() override;
    atb::Status InferShape(
        const atb::SVector<atb::TensorDesc> &inTensorDesc,
        atb::SVector<atb::TensorDesc> &outTensorDesc) const override;
    [[nodiscard]] uint32_t GetInputNum() const override;
    [[nodiscard]] uint32_t GetOutputNum() const override;

protected:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;
    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;

private:
    AclNNMoeGatingTopKParam param_;
};
}  // namespace atb_speed::common

#endif

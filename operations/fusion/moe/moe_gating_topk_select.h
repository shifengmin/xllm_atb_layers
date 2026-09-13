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

#ifndef ATB_SPEED_MODELS_MOE_GATING_TOPK_SELECT_H
#define ATB_SPEED_MODELS_MOE_GATING_TOPK_SELECT_H

#include <cstdint>
#include <string>

namespace atb_speed {
namespace common {

constexpr int32_t FUSED_ADD_TOPK_SORT_UNIT = 32;
constexpr int64_t MOE_GATING_TOPK_GROUP_SELECT_MODE = 1;
constexpr int64_t MOE_GATING_TOPK_RENORM = 1;
constexpr int64_t MOE_GATING_TOPK_NORM_TYPE = 1;
constexpr double MOE_GATING_TOPK_EPS = 1e-20;

// aclnnMoeFusedAddTopk tiles experts as n_group * 32. n_group<=1 (GLM-5.x)
// or experts beyond that tile width cannot use it; noAuxTc then uses
// aclnnMoeGatingTopK instead.
inline bool UseMoeGatingTopK(bool enableFusedTopk,
    const std::string &routingMethod, int32_t numOfGroups, int32_t numOfExperts)
{
    const bool fusedAddtopkIllegal =
        numOfGroups <= 1 ||
        numOfExperts > numOfGroups * FUSED_ADD_TOPK_SORT_UNIT;
    return enableFusedTopk && routingMethod == "noAuxTc" && fusedAddtopkIllegal;
}

}  // namespace common
}  // namespace atb_speed

#endif

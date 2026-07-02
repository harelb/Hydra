#include "hydra/frontend/subkeyframe_anchor.h"

#include <cstdint>
#include <limits>

namespace hydra {

std::optional<size_t> selectNearestAnchor(
    const std::vector<AnchorCandidate>& anchors,
    uint64_t subframe_ts_ns,
    uint64_t max_dt_ns) {
  std::optional<size_t> best;
  uint64_t best_dt = std::numeric_limits<uint64_t>::max();
  for (size_t i = 0; i < anchors.size(); ++i) {
    const uint64_t a = anchors[i].timestamp_ns;
    const uint64_t dt = a > subframe_ts_ns ? a - subframe_ts_ns : subframe_ts_ns - a;
    if (dt <= max_dt_ns && dt < best_dt) {
      best_dt = dt;
      best = i;
    }
  }
  return best;
}

Eigen::Isometry3d computeRelativeTransform(
    const Eigen::Isometry3d& world_T_anchor,
    const Eigen::Isometry3d& world_T_subframe) {
  return world_T_anchor.inverse() * world_T_subframe;
}

}  // namespace hydra

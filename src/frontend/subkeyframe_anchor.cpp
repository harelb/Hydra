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

std::unique_ptr<spark_dsg::SubKeyframeNodeAttributes> buildSubKeyframeAttrs(
    spark_dsg::NodeId anchor_id,
    const Eigen::Isometry3d& world_T_anchor,
    const Eigen::Isometry3d& world_T_subframe,
    uint64_t timestamp_ns,
    const std::string& image_folder) {
  auto attrs = std::make_unique<spark_dsg::SubKeyframeNodeAttributes>();
  attrs->anchor_node_id = anchor_id;
  const Eigen::Isometry3d rel =
      computeRelativeTransform(world_T_anchor, world_T_subframe);
  attrs->anchor_t_subframe = rel.translation();
  attrs->anchor_R_subframe = Eigen::Quaterniond(rel.rotation());
  attrs->position = world_T_subframe.translation();  // seed; backend refines
  attrs->image_folder = image_folder;
  attrs->timestamp = std::chrono::nanoseconds(timestamp_ns);
  return attrs;
}

}  // namespace hydra

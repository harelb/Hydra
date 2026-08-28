#include "hydra/frontend/subkeyframe_anchor.h"

#include <cstdint>
#include <cstdlib>
#include <limits>

namespace hydra {

std::optional<size_t> selectNearestAnchor(
    const std::vector<AnchorCandidate>& anchors,
    uint64_t subframe_ts_ns,
    const Eigen::Vector3d& subframe_position,
    double max_dist_m) {
  if (anchors.empty()) {
    return std::nullopt;
  }

  // Find the temporally-nearest anchor (ties -> lowest index).
  size_t best = 0;
  uint64_t best_dt = std::numeric_limits<uint64_t>::max();
  for (size_t i = 0; i < anchors.size(); ++i) {
    const auto a_ts = static_cast<int64_t>(anchors[i].timestamp_ns);
    const auto sub_ts = static_cast<int64_t>(subframe_ts_ns);
    const uint64_t dt = static_cast<uint64_t>(std::llabs(a_ts - sub_ts));
    if (dt < best_dt) {
      best_dt = dt;
      best = i;
    }
  }

  // Gate acceptance on the spatial distance of that same (temporally-nearest)
  // anchor. Do not fall back to a spatially-closer but temporally-farther one.
  const double dist =
      (anchors[best].world_T_anchor.translation() - subframe_position).norm();
  if (dist <= max_dist_m) {
    return best;
  }
  return std::nullopt;
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

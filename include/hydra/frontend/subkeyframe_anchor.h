#pragma once

#include <Eigen/Geometry>
#include <optional>
#include <vector>

#include "spark_dsg/scene_graph_types.h"

namespace hydra {

struct AnchorCandidate {
  spark_dsg::NodeId id;
  uint64_t timestamp_ns;
  Eigen::Isometry3d world_T_anchor;
};

std::optional<size_t> selectNearestAnchor(
    const std::vector<AnchorCandidate>& anchors,
    uint64_t subframe_ts_ns,
    uint64_t max_dt_ns);

Eigen::Isometry3d computeRelativeTransform(
    const Eigen::Isometry3d& world_T_anchor,
    const Eigen::Isometry3d& world_T_subframe);

}  // namespace hydra

#pragma once

#include <Eigen/Geometry>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "spark_dsg/node_attributes.h"
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

// Builds a sub-keyframe node's attributes. The relative transform
// anchor_T_subframe is the durable source of truth; the world position is
// seeded from world_T_subframe and refined by the backend later.
std::unique_ptr<spark_dsg::SubKeyframeNodeAttributes> buildSubKeyframeAttrs(
    spark_dsg::NodeId anchor_id,
    const Eigen::Isometry3d& world_T_anchor,
    const Eigen::Isometry3d& world_T_subframe,
    uint64_t timestamp_ns,
    const std::string& image_folder);

}  // namespace hydra

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

// Handed off from the SubKeyframeModule thread to the frontend (GraphBuilder)
// thread, which is the only thread allowed to mutate the frontend DSG. Carries
// everything needed to build a sub-keyframe node once anchors exist.
struct SubKeyframeRequest {
  uint64_t timestamp_ns;
  Eigen::Isometry3d world_T_subframe;  // world_T_body of the sub-keyframe
  std::string image_folder;
};

// Selects the TEMPORALLY-nearest anchor (time is monotonic along the
// trajectory, so this stays within the same pass even across loop closures),
// then gates acceptance on SPATIAL distance: the rigid anchor_T_subframe
// transform's error grows with the anchor<->sub-keyframe spatial span, and a
// distance bound also correctly accepts post-stop sub-keyframes whose
// temporally-nearest anchor is time-far but spatially co-located. Does NOT
// fall back to a spatially-nearer but temporally-farther anchor.
std::optional<size_t> selectNearestAnchor(
    const std::vector<AnchorCandidate>& anchors,
    uint64_t subframe_ts_ns,
    const Eigen::Vector3d& subframe_position,
    double max_dist_m);

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

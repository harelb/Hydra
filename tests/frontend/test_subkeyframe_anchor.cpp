#include <gtest/gtest.h>

#include "hydra/frontend/subkeyframe_anchor.h"

namespace hydra {

namespace {
Eigen::Isometry3d atPosition(double x, double y, double z) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(x, y, z);
  return pose;
}
}  // namespace

TEST(SubkeyframeAnchor, TemporalSelectionSpatialGate) {
  // Temporally-nearest to ts=2100 is the t=2000 anchor. Place it within
  // max_dist so it is accepted, even though other anchors are spatially
  // closer to the sub-keyframe position.
  std::vector<AnchorCandidate> anchors = {
      {1, 1000, atPosition(0, 0, 0)},
      {2, 2000, atPosition(1, 0, 0)},
      {3, 3000, atPosition(0, 0, 0)},
  };
  const Eigen::Vector3d subframe_position(1.1, 0, 0);
  auto idx = selectNearestAnchor(anchors, 2100, subframe_position, /*max_dist_m=*/2.0);
  ASSERT_TRUE(idx.has_value());
  EXPECT_EQ(anchors[*idx].id, 2u);
}

TEST(SubkeyframeAnchor, RejectsWhenTemporallyNearestIsFar) {
  // Temporally-nearest anchor (t=2000) is spatially far from the sub-keyframe,
  // while a temporally-farther anchor (t=3000) is spatially close. Loop-safety
  // requires we reject rather than fall back to the spatially-closer one.
  std::vector<AnchorCandidate> anchors = {
      {1, 1000, atPosition(0, 0, 0)},
      {2, 2000, atPosition(100, 0, 0)},
      {3, 3000, atPosition(0, 0, 0)},
  };
  const Eigen::Vector3d subframe_position(0, 0, 0);
  auto idx = selectNearestAnchor(anchors, 2100, subframe_position, /*max_dist_m=*/2.0);
  EXPECT_FALSE(idx.has_value());
}

TEST(SubkeyframeAnchor, AcceptsPostStopNearAnchor) {
  // Temporally-nearest anchor is far in time (large dt) but spatially at the
  // sub-keyframe (dist ~ 0), e.g. the robot stopped. Should be accepted.
  std::vector<AnchorCandidate> anchors = {
      {1, 1000, atPosition(0, 0, 0)},
  };
  const Eigen::Vector3d subframe_position(0.01, 0, 0);
  auto idx =
      selectNearestAnchor(anchors, 9000000000ULL, subframe_position, /*max_dist_m=*/2.0);
  ASSERT_TRUE(idx.has_value());
  EXPECT_EQ(anchors[*idx].id, 1u);
}

TEST(SubkeyframeAnchor, RelativeTransformComposesBack) {
  Eigen::Isometry3d world_T_anchor = Eigen::Isometry3d::Identity();
  world_T_anchor.translation() = Eigen::Vector3d(1, 0, 0);
  Eigen::Isometry3d world_T_sub = Eigen::Isometry3d::Identity();
  world_T_sub.translation() = Eigen::Vector3d(1.5, 0, 0);

  auto rel = computeRelativeTransform(world_T_anchor, world_T_sub);
  EXPECT_TRUE((world_T_anchor * rel).isApprox(world_T_sub));
  EXPECT_NEAR(rel.translation().x(), 0.5, 1e-9);
}

TEST(SubkeyframeAnchor, BuildsAttrsWithWorldPositionFromAnchor) {
  Eigen::Isometry3d world_T_anchor = Eigen::Isometry3d::Identity();
  world_T_anchor.translation() = Eigen::Vector3d(2, 0, 0);
  Eigen::Isometry3d world_T_sub = Eigen::Isometry3d::Identity();
  world_T_sub.translation() = Eigen::Vector3d(2.3, 0, 0);

  auto attrs = buildSubKeyframeAttrs(/*anchor_id=*/5u, world_T_anchor, world_T_sub,
                                     /*ts_ns=*/1234, "/data/subkf_1234");
  EXPECT_EQ(attrs->anchor_node_id, 5u);
  EXPECT_EQ(attrs->image_folder, "/data/subkf_1234");
  EXPECT_EQ(attrs->timestamp.count(), 1234);
  // initial world position seeded from world_T_sub (refined by backend later)
  EXPECT_TRUE(attrs->position.isApprox(Eigen::Vector3d(2.3, 0, 0)));
  // anchor_t_subframe = anchor^-1 * sub = (0.3, 0, 0)
  EXPECT_NEAR(attrs->anchor_t_subframe.x(), 0.3, 1e-9);
}

}  // namespace hydra

#include <gtest/gtest.h>

#include "hydra/frontend/subkeyframe_anchor.h"

namespace hydra {

TEST(SubkeyframeAnchor, SelectsNearestWithinTolerance) {
  std::vector<AnchorCandidate> anchors = {
      {1, 1000, Eigen::Isometry3d::Identity()},
      {2, 2000, Eigen::Isometry3d::Identity()},
      {3, 3000, Eigen::Isometry3d::Identity()},
  };
  auto idx = selectNearestAnchor(anchors, 2100, /*max_dt_ns=*/500);
  ASSERT_TRUE(idx.has_value());
  EXPECT_EQ(anchors[*idx].id, 2u);
}

TEST(SubkeyframeAnchor, RejectsWhenOutsideTolerance) {
  std::vector<AnchorCandidate> anchors = {{1, 1000, Eigen::Isometry3d::Identity()}};
  auto idx = selectNearestAnchor(anchors, 9000, /*max_dt_ns=*/500);
  EXPECT_FALSE(idx.has_value());
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

}  // namespace hydra

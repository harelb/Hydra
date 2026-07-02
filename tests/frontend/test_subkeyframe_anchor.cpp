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

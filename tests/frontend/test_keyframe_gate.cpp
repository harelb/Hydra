#include <gtest/gtest.h>
#include "hydra/frontend/keyframe_gate.h"

namespace hydra {

TEST(KeyframeGate, FirstCallAlwaysTriggers) {
  KeyframeGate gate({0.5, 15.0});
  EXPECT_TRUE(gate.shouldTrigger(Eigen::Vector3d(0, 0, 0),
                                 Eigen::Quaterniond::Identity()));
}

TEST(KeyframeGate, SmallMotionDoesNotTrigger) {
  KeyframeGate gate({0.5, 15.0});
  gate.shouldTrigger(Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
  EXPECT_FALSE(gate.shouldTrigger(Eigen::Vector3d(0.1, 0, 0),
                                  Eigen::Quaterniond::Identity()));
}

TEST(KeyframeGate, TranslationOverThresholdTriggers) {
  KeyframeGate gate({0.5, 15.0});
  gate.shouldTrigger(Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
  EXPECT_TRUE(gate.shouldTrigger(Eigen::Vector3d(0.6, 0, 0),
                                 Eigen::Quaterniond::Identity()));
}

TEST(KeyframeGate, RotationOverThresholdTriggers) {
  KeyframeGate gate({10.0, 15.0});  // large translation thresh so only rotation matters
  gate.shouldTrigger(Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
  const Eigen::Quaterniond r(
      Eigen::AngleAxisd(20.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ()));
  EXPECT_TRUE(gate.shouldTrigger(Eigen::Vector3d(0, 0, 0), r));
}

TEST(KeyframeGate, StateAdvancesOnlyOnTrigger) {
  KeyframeGate gate({0.5, 90.0});
  gate.shouldTrigger(Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
  // 0.3 then 0.3 again: neither individually >=0.5 from a non-advancing anchor,
  // but cumulative from the origin anchor the second (0.6) must trigger.
  EXPECT_FALSE(gate.shouldTrigger(Eigen::Vector3d(0.3, 0, 0),
                                  Eigen::Quaterniond::Identity()));
  EXPECT_TRUE(gate.shouldTrigger(Eigen::Vector3d(0.6, 0, 0),
                                 Eigen::Quaterniond::Identity()));
}

}  // namespace hydra

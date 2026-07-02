#pragma once

#include <Eigen/Geometry>

namespace hydra {

// Pure translation/rotation keyframe trigger. No ROS, no DSG. Mirrors the
// threshold logic in AgentImageExtractor but is independently testable and
// reusable by the full-rate sub-keyframe module.
class KeyframeGate {
 public:
  struct Config {
    double min_translation_m = 0.25;
    double min_rotation_deg = 15.0;
  };

  explicit KeyframeGate(const Config& config) : config_(config) {}

  bool shouldTrigger(const Eigen::Vector3d& position,
                     const Eigen::Quaterniond& orientation);

 private:
  Config config_;
  bool initialized_ = false;
  Eigen::Vector3d last_position_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond last_orientation_ = Eigen::Quaterniond::Identity();
};

}  // namespace hydra

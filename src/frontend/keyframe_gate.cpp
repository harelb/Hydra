#include "hydra/frontend/keyframe_gate.h"

#include <config_utilities/config.h>

namespace hydra {

void declare_config(KeyframeGate::Config& config) {
  using namespace config;
  name("KeyframeGate::Config");
  field(config.min_translation_m, "min_translation_m");
  field(config.min_rotation_deg, "min_rotation_deg");
}

bool KeyframeGate::shouldTrigger(const Eigen::Vector3d& position,
                                 const Eigen::Quaterniond& orientation) {
  bool trigger = false;
  if (!initialized_) {
    trigger = true;
  } else {
    const double translation_diff = (position - last_position_).norm();
    const double angular_diff =
        last_orientation_.angularDistance(orientation) * 180.0 / M_PI;
    if (translation_diff >= config_.min_translation_m ||
        angular_diff >= config_.min_rotation_deg) {
      trigger = true;
    }
  }

  if (trigger) {
    last_position_ = position;
    last_orientation_ = orientation;
    initialized_ = true;
  }
  return trigger;
}

}  // namespace hydra

#include "hydra_ros/frontend/sub_keyframe_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <glog/logging.h>

namespace hydra {

void declare_config(KeyframeGate::Config& config) {
  using namespace config;
  name("KeyframeGate::Config");
  field(config.min_translation_m, "min_translation_m");
  field(config.min_rotation_deg, "min_rotation_deg");
}

void declare_config(SubKeyframeModule::Config& config) {
  using namespace config;
  name("SubKeyframeModule::Config");
  field(config.enabled, "enabled");
  field(config.image_output_path, "image_output_path");
  field(config.gate, "gate");
  field(config.receiver, "receiver");
  field(config.tf_lookup, "tf_lookup");
}

SubKeyframeModule::SubKeyframeModule(const Config& config,
                                     const SharedDsgInfo::Ptr& dsg)
    : config_(config), dsg_(dsg), gate_(config.gate) {
  if (config_.enabled && !config_.image_output_path.empty()) {
    writer_ = std::make_unique<KeyframeWriter>(config_.image_output_path);
  }
}

SubKeyframeModule::~SubKeyframeModule() { stop(); }

void SubKeyframeModule::start() {
  if (!config_.enabled) {
    return;
  }

  receiver_ = std::make_unique<RGBDImageReceiver>(config_.receiver, "subkf");
  receiver_->init();
  lookup_ = std::make_unique<TFLookup>(config_.tf_lookup);
  should_shutdown_ = false;
  thread_ = std::make_unique<std::thread>(&SubKeyframeModule::spin, this);
}

void SubKeyframeModule::stop() {
  should_shutdown_ = true;
  if (thread_) {
    thread_->join();
    thread_.reset();
  }
}

std::string SubKeyframeModule::printInfo() const {
  return config::toString(config_);
}

void SubKeyframeModule::spin() {
  while (!should_shutdown_) {
    if (!receiver_->queue.poll()) {
      continue;
    }

    // The queue holds SensorInputPacket::Ptr; the RGBD receiver always pushes
    // ImageInputPacket, which carries color + depth.
    const auto base_packet = receiver_->queue.pop();
    const auto packet = std::dynamic_pointer_cast<ImageInputPacket>(base_packet);
    if (!packet) {
      continue;
    }

    const auto pose = lookup_->getBodyPose(packet->timestamp_ns);
    if (!pose) {
      continue;
    }
    const Eigen::Isometry3d world_T_body = pose.target_T_source();

    if (!gate_.shouldTrigger(world_T_body.translation(),
                             Eigen::Quaterniond(world_T_body.rotation()))) {
      continue;
    }

    // TODO(Phase 3): write camera calibration once via writer_->writeCalib().
    // RGBDImageReceiver exposes no Sensor object, so fx/fy/cx/cy are not
    // available here; calib persistence needs a CameraInfo subscription or a
    // sensor config field. Deferred rather than fabricating intrinsics.
    if (writer_) {
      writer_->write(
          packet->timestamp_ns, packet->color, packet->depth, world_T_body);
    }
  }
}

}  // namespace hydra

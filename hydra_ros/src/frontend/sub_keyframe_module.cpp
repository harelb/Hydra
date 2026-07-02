#include "hydra_ros/frontend/sub_keyframe_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <glog/logging.h>

#include "hydra/common/global_info.h"
#include "hydra/input/camera.h"

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
  field(config.sensor_name, "sensor_name");
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

    // Write the run-level camera calibration once, sourced from the globally
    // registered Camera sensor (intrinsics + extrinsics are constant for a
    // fixed camera, so they live outside the per-keyframe metadata). Mirrors
    // AgentImageExtractor::updateGraph.
    if (writer_ && !calib_written_) {
      const auto sensor = GlobalInfo::instance().getSensor(config_.sensor_name);
      const auto* camera = dynamic_cast<const Camera*>(sensor.get());
      if (camera) {
        const auto& cc = camera->getConfig();
        CameraCalib calib;
        calib.fx = cc.fx;
        calib.fy = cc.fy;
        calib.cx = cc.cx;
        calib.cy = cc.cy;
        calib.width = cc.width;
        calib.height = cc.height;
        calib.body_T_sensor = camera->body_T_sensor();
        writer_->writeCalib(calib);
        calib_written_ = true;
      } else {
        VLOG(1) << "[SubKeyframeModule] sensor '" << config_.sensor_name
                << "' not a Camera yet; calib deferred to a later frame";
      }
    }

    if (writer_) {
      writer_->write(packet->timestamp_ns, packet->color, packet->depth);
    }
  }
}

}  // namespace hydra

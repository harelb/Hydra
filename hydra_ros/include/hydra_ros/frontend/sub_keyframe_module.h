#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "hydra/common/module.h"
#include "hydra/common/shared_dsg_info.h"
#include "hydra/frontend/keyframe_gate.h"
#include "hydra/frontend/keyframe_writer.h"
#include "hydra_ros/input/image_receiver.h"
#include "hydra_ros/utils/tf_lookup.h"

namespace hydra {

// Standalone full-rate keyframe capture. Owns its own RGBD receiver (no label
// dependency) and TF lookup, so it is NOT gated by semantic_inference. Writes
// RGB+depth+pose to disk. DSG-node/anchor association is added in Phase 3.
class SubKeyframeModule : public Module {
 public:
  struct Config {
    bool enabled = false;
    std::string image_output_path;
    std::string sensor_name = "camera";
    KeyframeGate::Config gate;
    RGBDImageReceiver::Config receiver;
    TFLookup::Config tf_lookup;
  };

  SubKeyframeModule(const Config& config, const SharedDsgInfo::Ptr& dsg);
  ~SubKeyframeModule();

  void start() override;
  void stop() override;
  std::string printInfo() const override;

 private:
  void spin();

  Config config_;
  SharedDsgInfo::Ptr dsg_;
  std::unique_ptr<RGBDImageReceiver> receiver_;
  std::unique_ptr<TFLookup> lookup_;
  KeyframeGate gate_;
  std::unique_ptr<KeyframeWriter> writer_;
  bool calib_written_ = false;
  std::atomic<bool> should_shutdown_{false};
  std::unique_ptr<std::thread> thread_;
};

// KeyframeGate::Config (defined in hydra) has no declare_config of its own; we
// provide one here so it can be parsed as a nested field of the module config.
void declare_config(KeyframeGate::Config& config);
void declare_config(SubKeyframeModule::Config& config);

}  // namespace hydra

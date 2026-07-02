#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include "hydra/common/module.h"
#include "hydra/common/shared_dsg_info.h"
#include "hydra/frontend/keyframe_gate.h"
#include "hydra/frontend/keyframe_writer.h"
#include "hydra_ros/utils/tf_lookup.h"

namespace hydra {

// Full-rate keyframe capture. Drains the shared PipelineQueues::subkeyframe_queue
// (filled by the image receiver's in-memory tap) rather than owning its own RGBD
// subscription, so there is one wire-decode per topic. Adds pose via TF lookup,
// so it is NOT gated by semantic_inference. Writes RGB+depth+pose to disk.
// DSG-node/anchor association is added in Phase 3.
class SubKeyframeModule : public Module {
 public:
  struct Config {
    bool enabled = false;
    std::string image_output_path;
    std::string sensor_name = "camera";
    KeyframeGate::Config gate;
    TFLookup::Config tf_lookup;
    size_t queue_max_size = 30;
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
  std::unique_ptr<TFLookup> lookup_;
  KeyframeGate gate_;
  std::unique_ptr<KeyframeWriter> writer_;
  bool calib_written_ = false;
  std::atomic<bool> should_shutdown_{false};
  std::unique_ptr<std::thread> thread_;
};

void declare_config(SubKeyframeModule::Config& config);

}  // namespace hydra

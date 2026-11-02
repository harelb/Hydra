/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#pragma once

#include <hydra/common/module.h>
#include <hydra/frontend/keyframe_selector.h>
#include <hydra/utils/logging.h>

#include <atomic>
#include <list>
#include <memory>
#include <string>
#include <thread>

#include "hydra_ros/utils/tf_lookup.h"

namespace hydra {

class ImageStore;
struct SubKeyframeInput;

/**
 * @brief Captures sub-keyframes from the full-rate color and depth images.
 *
 * Drains PipelineQueues::subkeyframe_queue (filled by the image receivers, so it is
 * independent of the rate of the semantic input), looks up the body pose and writes
 * images accepted by the keyframe policy (the frame rate knob) to the image store as
 * sub-keyframe frames with their odometric pose. The frontend later anchors them to a
 * nearby agent node (see GraphBuilder::anchorPendingFrames). Disabled if no image store
 * is configured (see GlobalInfo::imageStore). Images queued when the module is stopped
 * are still processed.
 */
class SubKeyframeModule : public Module {
 public:
  struct Config : public VerbosityConfig {
    Config();

    //! Name of the sensor to capture sub-keyframes from (empty accepts any sensor)
    std::string sensor_name;
    //! Policy deciding which images become sub-keyframes. It is given the timestamp
    //! and body pose of the image and of the last sub-keyframe (no images). Default:
    //! DistancePolicy with min_pose_separation 0.25, rotation_separation_weight 0 and
    //! no time limit, i.e., a sub-keyframe every 0.25 m of translation
    config::VirtualConfig<KeyframePolicy> keyframe_policy;
    //! Body pose lookup
    TFLookup::Config tf_lookup;
    //! Maximum number of pending images (excess is dropped)
    size_t queue_max_size = 30;
  } const config;

  explicit SubKeyframeModule(const Config& config);

  virtual ~SubKeyframeModule();

  void start() override;

  void stop() override;

  std::string printInfo() const override;

  //! Whether sub-keyframes are captured (false if no image store is configured or it
  //! could not be opened)
  bool enabled() const { return store_ != nullptr; }

  //! @brief Process one input (exposed for testing)
  bool processInput(const SubKeyframeInput& input,
                    const Eigen::Isometry3d& world_T_body);

 private:
  void spin();

  //! Process the next queued image (the queue must not be empty)
  void processNext();

  void stopImpl();

  //! Whether the body pose is a new sub-keyframe (recorded as the last one if so)
  bool isKeyframe(uint64_t timestamp_ns, const Eigen::Isometry3d& world_T_body);

  std::unique_ptr<KeyframePolicy> policy_;
  //! Timestamp and pose (no images) of the last sub-keyframe, if any
  std::list<InputData::ConstPtr> last_keyframe_;
  std::shared_ptr<ImageStore> store_;
  std::unique_ptr<TFLookup> lookup_;
  std::atomic<bool> should_shutdown_{false};
  std::unique_ptr<std::thread> spin_thread_;
};

void declare_config(SubKeyframeModule::Config& config);

}  // namespace hydra

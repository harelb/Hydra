#include "hydra_ros/frontend/sub_keyframe_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <glog/logging.h>

#include "hydra/common/global_info.h"
#include "hydra/common/pipeline_queues.h"
#include "hydra/frontend/subkeyframe_anchor.h"
#include "hydra/input/camera.h"

namespace hydra {

void declare_config(SubKeyframeModule::Config& config) {
  using namespace config;
  name("SubKeyframeModule::Config");
  field(config.enabled, "enabled");
  field(config.image_output_path, "image_output_path");
  field(config.sensor_name, "sensor_name");
  field(config.gate, "gate");
  field(config.tf_lookup, "tf_lookup");
  field(config.queue_max_size, "queue_max_size");
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
  // M-2: disabled unless explicitly enabled AND an output path is configured.
  if (!config_.enabled || config_.image_output_path.empty()) {
    return;
  }

  // Create the shared tap queue. Its mere presence enables the image receiver's
  // in-memory tap (which no-ops while the queue is null). M-1: bound the queue
  // so a stalled writer sheds frames instead of growing without limit.
  auto queue = std::make_shared<MessageQueue<SensorInputPacket::Ptr>>();
  queue->max_size = config_.queue_max_size;
  PipelineQueues::instance().subkeyframe_queue = queue;

  // Hand-off queue drained by GraphBuilder on the frontend thread. Bounded so a
  // stalled frontend sheds sub-keyframe requests instead of growing unbounded.
  auto node_queue = std::make_shared<MessageQueue<SubKeyframeRequest>>();
  node_queue->max_size = config_.queue_max_size;
  PipelineQueues::instance().subkeyframe_node_queue = node_queue;

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
  // Drop the shared queue so the image receiver's tap stops pushing frames.
  PipelineQueues::instance().subkeyframe_queue.reset();
  PipelineQueues::instance().subkeyframe_node_queue.reset();
}

std::string SubKeyframeModule::printInfo() const {
  return config::toString(config_);
}

void SubKeyframeModule::spin() {
  while (!should_shutdown_) {
    // I-1: guard the whole loop body so one malformed frame (bad TF, decode,
    // or write error) drops that frame instead of tearing down the thread.
    try {
      auto& queue = PipelineQueues::instance().subkeyframe_queue;
      if (!queue || !queue->poll()) {
        continue;
      }

      // The queue holds SensorInputPacket::Ptr; the image receiver tap pushes
      // ImageInputPacket, which carries color + depth.
      const auto base_packet = queue->pop();
      const auto packet =
          std::dynamic_pointer_cast<ImageInputPacket>(base_packet);
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

      std::string image_folder;
      if (writer_) {
        writer_->write(packet->timestamp_ns, packet->color, packet->depth);
        image_folder = config_.image_output_path + "/subkf_" +
                       std::to_string(packet->timestamp_ns);
      }

      // Hand the request off to the frontend (GraphBuilder) thread, which owns
      // all mutation of the frontend DSG. Doing anchor association + node
      // creation here would race GraphBuilder's unlocked reads/mergeGraph of the
      // same graph. Non-blocking push: if the queue is full we shed this frame.
      if (PipelineQueues::instance().subkeyframe_node_queue) {
        PipelineQueues::instance().subkeyframe_node_queue->push(
            SubKeyframeRequest{packet->timestamp_ns, world_T_body, image_folder},
            /*blocking=*/false);
      }
    } catch (const std::exception& e) {
      LOG_EVERY_N(WARNING, 100)
          << "[SubKeyframeModule] frame dropped: " << e.what();
    }
  }
}

}  // namespace hydra

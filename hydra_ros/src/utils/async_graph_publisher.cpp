#include "hydra_ros/utils/async_graph_publisher.h"

namespace hydra {

AsyncGraphPublisher::AsyncGraphPublisher(SendFunc send)
    : send_(std::move(send)), worker_([this] { run(); }) {}

AsyncGraphPublisher::~AsyncGraphPublisher() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    should_stop_ = true;
    pending_.reset();  // drop any pending snapshot
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void AsyncGraphPublisher::submit(std::shared_ptr<spark_dsg::DynamicSceneGraph> snapshot,
                                 uint64_t timestamp_ns) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_) {
      ++dropped_;  // latest wins
    }
    pending_ = std::move(snapshot);
    pending_ts_ = timestamp_ns;
  }
  cv_.notify_all();
}

void AsyncGraphPublisher::run() {
  while (true) {
    std::shared_ptr<spark_dsg::DynamicSceneGraph> snapshot;
    uint64_t ts = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return should_stop_ || pending_; });
      if (should_stop_) {
        return;
      }
      snapshot = std::move(pending_);
      pending_.reset();
      ts = pending_ts_;
    }
    send_(*snapshot, ts);
  }
}

}  // namespace hydra

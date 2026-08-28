#pragma once
#include <spark_dsg/scene_graph.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace hydra {

// Serializes+publishes DSG snapshots on a worker thread with a depth-1 latest-wins
// slot, so the backend spin never blocks on serialization.
class AsyncGraphPublisher {
 public:
  using SendFunc = std::function<void(const spark_dsg::SceneGraph&, uint64_t)>;

  explicit AsyncGraphPublisher(SendFunc send);
  ~AsyncGraphPublisher();

  void submit(std::shared_ptr<spark_dsg::SceneGraph> snapshot,
              uint64_t timestamp_ns);

  size_t numDropped() const { return dropped_; }

 private:
  void run();

  SendFunc send_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::shared_ptr<spark_dsg::SceneGraph> pending_;
  uint64_t pending_ts_ = 0;
  bool should_stop_ = false;
  std::atomic<size_t> dropped_{0};
  std::thread worker_;
};

}  // namespace hydra

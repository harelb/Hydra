#include <gtest/gtest.h>
#include <hydra_ros/utils/async_graph_publisher.h>
#include <spark_dsg/dynamic_scene_graph.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace hydra {

namespace {

struct FakeSender {
  std::mutex m;
  std::condition_variable cv;
  std::vector<uint64_t> stamps;
  bool block = false;

  void send(const spark_dsg::DynamicSceneGraph&, uint64_t ts) {
    std::unique_lock<std::mutex> lock(m);
    while (block) {
      cv.wait_for(lock, std::chrono::milliseconds(10));
    }
    stamps.push_back(ts);
    cv.notify_all();
  }

  void waitForCount(size_t n) {
    std::unique_lock<std::mutex> lock(m);
    cv.wait_for(lock, std::chrono::seconds(5), [&] { return stamps.size() >= n; });
  }
};

std::shared_ptr<spark_dsg::DynamicSceneGraph> makeGraph() {
  return std::make_shared<spark_dsg::DynamicSceneGraph>();
}

}  // namespace

TEST(AsyncGraphPublisher, PublishesSubmittedSnapshot) {
  FakeSender sender;
  AsyncGraphPublisher pub(
      [&sender](const auto& graph, uint64_t ts) { sender.send(graph, ts); });
  pub.submit(makeGraph(), 42);
  sender.waitForCount(1);
  EXPECT_EQ(sender.stamps, std::vector<uint64_t>{42});
}

TEST(AsyncGraphPublisher, LatestWinsWhileWorkerBusy) {
  FakeSender sender;
  {
    std::lock_guard<std::mutex> lock(sender.m);
    sender.block = true;
  }
  AsyncGraphPublisher pub(
      [&sender](const auto& graph, uint64_t ts) { sender.send(graph, ts); });

  pub.submit(makeGraph(), 1);  // worker picks this up and blocks inside send
  // give the worker a moment to take snapshot 1
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  pub.submit(makeGraph(), 2);  // queued
  pub.submit(makeGraph(), 3);  // replaces 2 (latest wins)

  {
    std::lock_guard<std::mutex> lock(sender.m);
    sender.block = false;
  }
  sender.cv.notify_all();
  sender.waitForCount(2);

  EXPECT_EQ(sender.stamps, (std::vector<uint64_t>{1, 3}));
  EXPECT_EQ(pub.numDropped(), 1u);
}

TEST(AsyncGraphPublisher, CleanShutdownWithPendingSnapshot) {
  FakeSender sender;
  {
    auto pub = std::make_unique<AsyncGraphPublisher>(
        [&sender](const auto& graph, uint64_t ts) { sender.send(graph, ts); });
    pub->submit(makeGraph(), 7);
    // destructor must join without deadlock regardless of worker state
  }
  SUCCEED();
}

}  // namespace hydra

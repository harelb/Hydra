#include <gtest/gtest.h>

#include "hydra_ros/input/ros_input_module.h"

namespace hydra {

TEST(RosInputModuleConfig, SensorRemappingPreservesReceiverQueueLimit) {
  RosInputModule::Config config;
  config.max_receiver_queue_size = 5;
  const auto remapped = config.remapSensors();
  EXPECT_EQ(remapped.max_receiver_queue_size, 5u);
  EXPECT_TRUE(remapped.inputs.empty());
}

TEST(RosInputModuleConfig, ExplicitUnlimitedReceiverRemainsUnlimited) {
  RosInputModule::Config config;
  config.max_receiver_queue_size = 0;
  EXPECT_EQ(config.remapSensors().max_receiver_queue_size, 0u);
}

}  // namespace hydra

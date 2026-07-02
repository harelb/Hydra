#include <gtest/gtest.h>
#include <config_utilities/config.h>
#include <config_utilities/parsing/yaml.h>

#include "hydra_ros/frontend/sub_keyframe_module.h"

namespace hydra {

TEST(SubKeyframeModule, ConfigParsesFromYaml) {
  const std::string yaml = R"yaml(
enabled: true
image_output_path: /tmp/subkf
sensor_name: camera
gate: {min_translation_m: 0.25, min_rotation_deg: 15.0}
tf_lookup: {max_tries: 5}
queue_max_size: 30
)yaml";
  const auto node = YAML::Load(yaml);
  const auto config = config::fromYaml<SubKeyframeModule::Config>(node);
  EXPECT_TRUE(config.enabled);
  EXPECT_EQ(config.image_output_path, "/tmp/subkf");
  EXPECT_EQ(config.sensor_name, "camera");
  EXPECT_DOUBLE_EQ(config.gate.min_translation_m, 0.25);
  EXPECT_DOUBLE_EQ(config.gate.min_rotation_deg, 15.0);
  EXPECT_EQ(config.queue_max_size, 30u);
}

}  // namespace hydra

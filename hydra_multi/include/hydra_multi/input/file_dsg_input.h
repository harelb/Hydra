#pragma once

#include "hydra_multi/input/input.h"

namespace hydra_multi {

class FileDsgInput : public Input {
 public:
  struct Config {
    std::string dsg_json;
    bool force_robot_id = false;
    //! Session label added to the metadata of every node (empty disables)
    std::string session;
    //! Optional JSON {"<node id>": {metadata}} merged into the metadata of the robot's
    //! nodes (ids as in dsg_json), e.g. change annotations between sessions
    std::string node_metadata_path;
  } const config;

  FileDsgInput(const Config& config,
               UnitInterfaceState::Ptr state,
               std::string name,
               size_t id);

  ~FileDsgInput() = default;

  void init() override;

  void stop() override;

 private:
  void annotateNodes(spark_dsg::SceneGraph& graph) const;
};

void declare_config(FileDsgInput::Config& config);

}  // namespace hydra_multi

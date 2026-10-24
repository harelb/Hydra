#include "hydra_multi/input/file_gps_input.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/types/path.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>

#include <cctype>
#include <fstream>
#include <sstream>

namespace hydra_multi {
namespace {

static const auto registration_ =
    config::RegistrationWithConfig<Input,
                                   FileGpsInput,
                                   FileGpsInput::Config,
                                   UnitInterfaceState::Ptr,
                                   std::string,
                                   size_t>("FileGpsInput");

}

using config::Path;

void declare_config(FileGpsInput::Config& config) {
  using namespace config;
  name("FileGpsInput::Config");
  field<Path>(config.gps_path, "gps_path");
  field(config.body_t_gps, "body_t_gps");
  field(config.utm_zone, "utm_zone");
  field(config.min_status, "min_status");
  field(config.max_sigma_m, "max_sigma_m");

  check<Path::Exists>(config.gps_path, "gps_path");
  checkCondition(config.body_t_gps.size() == 3, "body_t_gps must have 3 entries");
  check(config.max_sigma_m, GT, 0.0, "max_sigma_m");
}

FileGpsInput::FileGpsInput(const Config& config,
                           UnitInterfaceState::Ptr state,
                           std::string name,
                           size_t id)
    : Input(state, name, id), config(config::checkValid(config)) {}

void FileGpsInput::init() {
  const auto measurements = loadGpsCsv(config.gps_path, config.utm_zone);
  int zone = config.utm_zone;
  if (zone == 0) {
    // recover the zone picked for the first fix
    std::ifstream file(config.gps_path);
    std::string line;
    while (zone == 0 && std::getline(file, line)) {
      std::stringstream ss(line);
      std::string stamp, lat, lon;
      if (line.empty() || !std::isdigit(line[0]) || !std::getline(ss, stamp, ',') ||
          !std::getline(ss, lat, ',') || !std::getline(ss, lon, ',')) {
        continue;
      }
      zone = latLonToUtm(std::stod(lat), std::stod(lon)).zone;
    }
  }

  const double max_variance = config.max_sigma_m * config.max_sigma_m;

  std::lock_guard<std::mutex> lock(state_->mutex);
  state_->gps_measurements.clear();
  for (const auto& meas : measurements) {
    if (meas.status < config.min_status) {
      continue;
    }

    if (std::max(meas.variance.x(), meas.variance.y()) > max_variance) {
      continue;
    }

    state_->gps_measurements.push_back(meas);
  }

  state_->body_t_gps << config.body_t_gps[0], config.body_t_gps[1], config.body_t_gps[2];
  state_->gps_utm_zone = zone;
  LOG(INFO) << "[" << name_ << "] kept " << state_->gps_measurements.size() << " / "
            << measurements.size() << " GPS fixes from " << config.gps_path;
}

void FileGpsInput::stop() {}

}  // namespace hydra_multi

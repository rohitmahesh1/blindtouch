#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "blindtouch_interfaces/msg/claw_command.hpp"
#include "blindtouch_interfaces/msg/simulation_state.hpp"
#include "blindtouch_interfaces/msg/tactile_state.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace blindtouch_control
{

using Action = std::array<float, 4>;
using Observation = std::array<float, 45>;
using PerFingerForces = std::array<float, 3>;

inline constexpr std::array<float, 4> kCtrlLow{-0.02F, 0.0F, 0.0F, 0.0F};
inline constexpr std::array<float, 4> kCtrlHigh{0.14F, 0.055F, 0.055F, 0.055F};
inline constexpr std::array<float, 4> kEffortScale{80.0F, 15.0F, 15.0F, 15.0F};

inline const std::array<std::string, 4> kActionNames{
  "palm_lift",
  "finger_1_close",
  "finger_2_close",
  "finger_3_close",
};

inline float clip(float value, float low, float high)
{
  return std::max(low, std::min(value, high));
}

inline Action clip_action(const Action & action)
{
  return {
    clip(action[0], -1.0F, 1.0F),
    clip(action[1], -1.0F, 1.0F),
    clip(action[2], -1.0F, 1.0F),
    clip(action[3], -1.0F, 1.0F),
  };
}

inline Action action_from_command(const blindtouch_interfaces::msg::ClawCommand & msg)
{
  return clip_action({msg.palm_lift, msg.finger_1_close, msg.finger_2_close, msg.finger_3_close});
}

inline blindtouch_interfaces::msg::ClawCommand command_from_action(const Action & action)
{
  const auto clipped = clip_action(action);
  blindtouch_interfaces::msg::ClawCommand msg;
  msg.palm_lift = clipped[0];
  msg.finger_1_close = clipped[1];
  msg.finger_2_close = clipped[2];
  msg.finger_3_close = clipped[3];
  return msg;
}

struct ObservationBuilder
{
  float tactile_force_scale{5.0F};
  float velocity_scale{0.5F};
  int max_episode_steps{120};
  std::array<float, 4> joint_positions{kCtrlLow};
  std::array<float, 4> joint_velocities{0.0F, 0.0F, 0.0F, 0.0F};
  std::array<float, 4> efforts{0.0F, 0.0F, 0.0F, 0.0F};
  std::array<float, 27> taxel_forces{};
  Action previous_action{0.0F, 0.0F, 0.0F, 0.0F};
  float phase_lift_allowed{0.0F};
  int step{0};
  int last_step{-1};
  bool has_joint_state{false};
  bool has_tactile_state{false};
  bool has_sim_state{false};

  bool ready() const { return has_tactile_state; }
  bool policy_ready() const { return has_tactile_state && has_joint_state && has_sim_state; }

  void update_joint_state(const sensor_msgs::msg::JointState & msg)
  {
    std::unordered_map<std::string, std::size_t> index_by_name;
    for (std::size_t i = 0; i < msg.name.size(); ++i) {
      index_by_name[msg.name[i]] = i;
    }

    for (std::size_t action_index = 0; action_index < kActionNames.size(); ++action_index) {
      const auto iter = index_by_name.find(kActionNames[action_index]);
      if (iter == index_by_name.end()) {
        continue;
      }
      const auto source_index = iter->second;
      if (source_index < msg.position.size()) {
        joint_positions[action_index] = static_cast<float>(msg.position[source_index]);
      }
      if (source_index < msg.velocity.size()) {
        joint_velocities[action_index] = static_cast<float>(msg.velocity[source_index]);
      }
      if (source_index < msg.effort.size()) {
        efforts[action_index] = static_cast<float>(msg.effort[source_index]);
      }
    }
    has_joint_state = true;
  }

  void update_tactile_state(const blindtouch_interfaces::msg::TactileState & msg)
  {
    std::copy(msg.taxel_forces.begin(), msg.taxel_forces.end(), taxel_forces.begin());
    has_tactile_state = true;
  }

  bool update_sim_state(const blindtouch_interfaces::msg::SimulationState & msg)
  {
    const auto current_step = static_cast<int>(msg.step);
    const bool reset_detected = last_step >= 0 && current_step < last_step;
    last_step = current_step;
    step = current_step;
    phase_lift_allowed = msg.phase == "lift" ? 1.0F : 0.0F;
    has_sim_state = true;
    return reset_detected;
  }

  void set_previous_action(const Action & action)
  {
    previous_action = clip_action(action);
  }

  Observation base_observation() const
  {
    Observation observation{};
    for (std::size_t i = 0; i < 4; ++i) {
      observation[i] = clip(
        2.0F * (joint_positions[i] - kCtrlLow[i]) / (kCtrlHigh[i] - kCtrlLow[i]) - 1.0F,
        -1.0F,
        1.0F);
      observation[4 + i] = clip(joint_velocities[i] / velocity_scale, -1.0F, 1.0F);
      observation[8 + i] = clip(efforts[i] / kEffortScale[i], -1.0F, 1.0F);
    }

    for (std::size_t i = 0; i < taxel_forces.size(); ++i) {
      observation[12 + i] = clip(taxel_forces[i] / tactile_force_scale, 0.0F, 1.0F);
    }

    for (std::size_t i = 0; i < previous_action.size(); ++i) {
      observation[39 + i] = previous_action[i];
    }

    observation[43] = phase_lift_allowed;
    observation[44] = clip(1.0F - static_cast<float>(step) / static_cast<float>(max_episode_steps), -1.0F, 1.0F);
    return observation;
  }

  PerFingerForces per_finger_max_taxel_force() const
  {
    PerFingerForces forces{0.0F, 0.0F, 0.0F};
    for (std::size_t finger = 0; finger < 3; ++finger) {
      float max_force = 0.0F;
      for (std::size_t cell = 0; cell < 9; ++cell) {
        max_force = std::max(max_force, taxel_forces[finger * 9 + cell]);
      }
      forces[finger] = max_force;
    }
    return forces;
  }
};

}  // namespace blindtouch_control

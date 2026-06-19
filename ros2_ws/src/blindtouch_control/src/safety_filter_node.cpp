#include <algorithm>
#include <array>
#include <memory>
#include <string>

#include "blindtouch_control/observation.hpp"
#include "blindtouch_interfaces/msg/safety_filter_state.hpp"
#include "rclcpp/rclcpp.hpp"

namespace blindtouch_control
{

struct SafetyConfig
{
  float soft_force{0.25F};
  float high_force{0.32F};
  float force_rate{0.075F};
  float release_action{-0.28F};
  float close_cap{0.0F};
  float high_force_lift_cap{0.0F};
  float precontact_lift_cap{0.0F};
  int min_lift_contacts{2};
  float ready_force{0.060F};
  int stable_steps_required{2};
  float contact_threshold{0.015F};
  float smooth_alpha{0.55F};
  float lift_cap{0.55F};
};

struct SafetyResult
{
  Action action{};
  Action raw_action{};
  Action residual{};
  bool intervened{false};
  float max_force{0.0F};
  float max_force_rate{0.0F};
  int contact_count{0};
  int ready_count{0};
  int stable_steps{0};
};

class SafetyFilterNode : public rclcpp::Node
{
public:
  SafetyFilterNode()
  : Node("blindtouch_safety_filter")
  {
    declare_parameters();
    configure();

    raw_sub_ = create_subscription<blindtouch_interfaces::msg::ClawCommand>(
      get_parameter("input_topic").as_string(),
      10,
      [this](const blindtouch_interfaces::msg::ClawCommand::SharedPtr msg) {
        filter_command(*msg);
      });
    tactile_sub_ = create_subscription<blindtouch_interfaces::msg::TactileState>(
      get_parameter("tactile_topic").as_string(),
      10,
      [this](const blindtouch_interfaces::msg::TactileState::SharedPtr msg) {
        builder_.update_tactile_state(*msg);
      });
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      get_parameter("joint_state_topic").as_string(),
      10,
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
        builder_.update_joint_state(*msg);
      });
    sim_sub_ = create_subscription<blindtouch_interfaces::msg::SimulationState>(
      get_parameter("sim_state_topic").as_string(),
      10,
      [this](const blindtouch_interfaces::msg::SimulationState::SharedPtr msg) {
        if (builder_.update_sim_state(*msg)) {
          reset_filter();
        }
      });

    command_pub_ = create_publisher<blindtouch_interfaces::msg::ClawCommand>(
      get_parameter("output_topic").as_string(),
      10);
    state_pub_ = create_publisher<blindtouch_interfaces::msg::SafetyFilterState>(
      get_parameter("state_topic").as_string(),
      10);

    RCLCPP_INFO(get_logger(), "C++ tactile safety filter guarding raw claw commands");
  }

private:
  void declare_parameters()
  {
    declare_parameter<std::string>("input_topic", "blindtouch/raw_command");
    declare_parameter<std::string>("output_topic", "blindtouch/command");
    declare_parameter<std::string>("state_topic", "blindtouch/safety_state");
    declare_parameter<std::string>("tactile_topic", "blindtouch/tactile");
    declare_parameter<std::string>("joint_state_topic", "blindtouch/joint_states");
    declare_parameter<std::string>("sim_state_topic", "blindtouch/sim_state");
    declare_parameter<double>("tactile_force_scale", 5.0);
    declare_parameter<double>("velocity_scale", 0.5);
    declare_parameter<int>("max_episode_steps", 120);
    declare_parameter<double>("soft_force", 0.25);
    declare_parameter<double>("high_force", 0.32);
    declare_parameter<double>("force_rate", 0.075);
    declare_parameter<double>("release_action", -0.28);
    declare_parameter<double>("close_cap", 0.0);
    declare_parameter<double>("high_force_lift_cap", 0.0);
    declare_parameter<double>("precontact_lift_cap", 0.0);
    declare_parameter<int>("min_lift_contacts", 2);
    declare_parameter<double>("ready_force", 0.060);
    declare_parameter<int>("stable_steps", 2);
    declare_parameter<double>("contact_threshold", 0.015);
    declare_parameter<double>("smooth_alpha", 0.55);
    declare_parameter<double>("lift_cap", 0.55);
  }

  void configure()
  {
    builder_.tactile_force_scale = static_cast<float>(get_parameter("tactile_force_scale").as_double());
    builder_.velocity_scale = static_cast<float>(get_parameter("velocity_scale").as_double());
    builder_.max_episode_steps = get_parameter("max_episode_steps").as_int();
    config_.soft_force = static_cast<float>(get_parameter("soft_force").as_double());
    config_.high_force = static_cast<float>(get_parameter("high_force").as_double());
    config_.force_rate = static_cast<float>(get_parameter("force_rate").as_double());
    config_.release_action = static_cast<float>(get_parameter("release_action").as_double());
    config_.close_cap = static_cast<float>(get_parameter("close_cap").as_double());
    config_.high_force_lift_cap = static_cast<float>(get_parameter("high_force_lift_cap").as_double());
    config_.precontact_lift_cap = static_cast<float>(get_parameter("precontact_lift_cap").as_double());
    config_.min_lift_contacts = get_parameter("min_lift_contacts").as_int();
    config_.ready_force = static_cast<float>(get_parameter("ready_force").as_double());
    config_.stable_steps_required = get_parameter("stable_steps").as_int();
    config_.contact_threshold = static_cast<float>(get_parameter("contact_threshold").as_double());
    config_.smooth_alpha = static_cast<float>(get_parameter("smooth_alpha").as_double());
    config_.lift_cap = static_cast<float>(get_parameter("lift_cap").as_double());
  }

  void reset_filter()
  {
    previous_forces_ = {0.0F, 0.0F, 0.0F};
    previous_action_ = {0.0F, 0.0F, 0.0F, 0.0F};
    stable_steps_ = 0;
  }

  void filter_command(const blindtouch_interfaces::msg::ClawCommand & msg)
  {
    const auto result = apply(action_from_command(msg));
    builder_.set_previous_action(result.action);

    auto command_msg = command_from_action(result.action);
    command_msg.header.stamp = now();
    command_pub_->publish(command_msg);

    blindtouch_interfaces::msg::SafetyFilterState state_msg;
    state_msg.header.stamp = command_msg.header.stamp;
    state_msg.intervened = result.intervened;
    state_msg.max_force = result.max_force;
    state_msg.max_force_rate = result.max_force_rate;
    state_msg.contact_count = static_cast<std::uint32_t>(result.contact_count);
    state_msg.ready_count = static_cast<std::uint32_t>(result.ready_count);
    state_msg.stable_steps = static_cast<std::uint32_t>(result.stable_steps);
    state_msg.raw_action = result.raw_action;
    state_msg.filtered_action = result.action;
    state_msg.residual = result.residual;
    state_pub_->publish(state_msg);
  }

  SafetyResult apply(const Action & raw_action)
  {
    SafetyResult result;
    result.raw_action = clip_action(raw_action);
    auto guarded = result.raw_action;
    const auto forces = builder_.per_finger_max_taxel_force();
    PerFingerForces force_rate{};

    for (std::size_t i = 0; i < forces.size(); ++i) {
      force_rate[i] = forces[i] - previous_forces_[i];
    }

    result.contact_count = count_forces_at_least(forces, config_.contact_threshold);
    result.ready_count = count_forces_at_least(forces, config_.ready_force);
    result.max_force = *std::max_element(forces.begin(), forces.end());
    result.max_force_rate = *std::max_element(force_rate.begin(), force_rate.end());

    const bool balanced =
      result.ready_count >= config_.min_lift_contacts && result.max_force < config_.high_force;
    stable_steps_ = balanced ? stable_steps_ + 1 : 0;

    for (std::size_t finger_index = 0; finger_index < forces.size(); ++finger_index) {
      const auto action_index = finger_index + 1;
      if (forces[finger_index] >= config_.high_force || force_rate[finger_index] >= config_.force_rate) {
        guarded[action_index] = std::min(guarded[action_index], config_.release_action);
      } else if (forces[finger_index] >= config_.soft_force) {
        guarded[action_index] = std::min(guarded[action_index], config_.close_cap);
      }
    }

    if (result.max_force >= config_.high_force || result.max_force_rate >= config_.force_rate) {
      guarded[0] = std::min(guarded[0], config_.high_force_lift_cap);
    } else if (result.contact_count < config_.min_lift_contacts ||
      stable_steps_ < config_.stable_steps_required)
    {
      guarded[0] = std::min(guarded[0], config_.precontact_lift_cap);
    }

    guarded[0] = std::min(guarded[0], config_.lift_cap);

    if (config_.smooth_alpha < 1.0F) {
      for (std::size_t i = 0; i < guarded.size(); ++i) {
        guarded[i] = previous_action_[i] + config_.smooth_alpha * (guarded[i] - previous_action_[i]);
      }
    }

    result.action = clip_action(guarded);
    for (std::size_t i = 0; i < result.action.size(); ++i) {
      result.residual[i] = result.action[i] - result.raw_action[i];
    }
    result.intervened = std::any_of(
      result.residual.begin(),
      result.residual.end(),
      [](float value) { return std::abs(value) > 1e-6F; });
    result.stable_steps = stable_steps_;

    previous_forces_ = forces;
    previous_action_ = result.action;
    return result;
  }

  static int count_forces_at_least(const PerFingerForces & forces, float threshold)
  {
    return static_cast<int>(std::count_if(
      forces.begin(),
      forces.end(),
      [threshold](float force) { return force >= threshold; }));
  }

  ObservationBuilder builder_;
  SafetyConfig config_;
  PerFingerForces previous_forces_{0.0F, 0.0F, 0.0F};
  Action previous_action_{0.0F, 0.0F, 0.0F, 0.0F};
  int stable_steps_{0};

  rclcpp::Subscription<blindtouch_interfaces::msg::ClawCommand>::SharedPtr raw_sub_;
  rclcpp::Subscription<blindtouch_interfaces::msg::TactileState>::SharedPtr tactile_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<blindtouch_interfaces::msg::SimulationState>::SharedPtr sim_sub_;
  rclcpp::Publisher<blindtouch_interfaces::msg::ClawCommand>::SharedPtr command_pub_;
  rclcpp::Publisher<blindtouch_interfaces::msg::SafetyFilterState>::SharedPtr state_pub_;
};

}  // namespace blindtouch_control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<blindtouch_control::SafetyFilterNode>());
  rclcpp::shutdown();
  return 0;
}

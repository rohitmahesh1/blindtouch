#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

#include "blindtouch_control/observation.hpp"
#include "rclcpp/rclcpp.hpp"

namespace blindtouch_control
{

enum class ControllerType
{
  Fixed,
  Threshold,
  SafeForce,
  ProbeThenLift,
};

class ScriptedControllerNode : public rclcpp::Node
{
public:
  ScriptedControllerNode()
  : Node("blindtouch_scripted_controller")
  {
    declare_parameters();
    configure();

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
          reset_controller();
        }
        if (!last_outcome_.empty() && msg->outcome.empty()) {
          reset_controller();
        }
        last_outcome_ = msg->outcome;
      });

    command_pub_ = create_publisher<blindtouch_interfaces::msg::ClawCommand>(
      get_parameter("output_topic").as_string(),
      10);

    const auto publish_rate_hz = get_parameter("publish_rate_hz").as_double();
    if (publish_rate_hz <= 0.0) {
      throw std::runtime_error("publish_rate_hz must be positive");
    }
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_hz),
      [this]() { publish_command(); });

    RCLCPP_INFO(
      get_logger(),
      "C++ scripted controller '%s' publishing raw commands",
      get_parameter("controller_type").as_string().c_str());
  }

private:
  void declare_parameters()
  {
    declare_parameter<std::string>("controller_type", "safe_force");
    declare_parameter<std::string>("output_topic", "blindtouch/raw_command");
    declare_parameter<std::string>("tactile_topic", "blindtouch/tactile");
    declare_parameter<std::string>("joint_state_topic", "blindtouch/joint_states");
    declare_parameter<std::string>("sim_state_topic", "blindtouch/sim_state");
    declare_parameter<double>("publish_rate_hz", 25.0);
    declare_parameter<double>("tactile_force_scale", 5.0);
    declare_parameter<double>("velocity_scale", 0.5);
    declare_parameter<int>("max_episode_steps", 120);
    declare_parameter<double>("target_force", 0.50);
    declare_parameter<double>("force_band", 0.14);
    declare_parameter<double>("force_target", 0.65);
    declare_parameter<double>("close_rate", 0.18);
    declare_parameter<double>("trim_close_rate", 0.08);
    declare_parameter<double>("release_rate", 0.06);
    declare_parameter<double>("lift_rate", 1.0);
    declare_parameter<int>("stable_steps_required", 8);
    declare_parameter<int>("max_probe_steps", 100);
    declare_parameter<int>("close_steps", 24);
    declare_parameter<int>("lift_step", 24);
  }

  void configure()
  {
    builder_.tactile_force_scale = static_cast<float>(get_parameter("tactile_force_scale").as_double());
    builder_.velocity_scale = static_cast<float>(get_parameter("velocity_scale").as_double());
    builder_.max_episode_steps = get_parameter("max_episode_steps").as_int();

    const auto controller_type = get_parameter("controller_type").as_string();
    if (controller_type == "fixed") {
      controller_type_ = ControllerType::Fixed;
    } else if (controller_type == "threshold") {
      controller_type_ = ControllerType::Threshold;
    } else if (controller_type == "probe_then_lift") {
      controller_type_ = ControllerType::ProbeThenLift;
    } else if (controller_type == "safe_force") {
      controller_type_ = ControllerType::SafeForce;
    } else {
      throw std::runtime_error("unsupported controller_type: " + controller_type);
    }

    target_force_ = static_cast<float>(get_parameter("target_force").as_double());
    force_band_ = static_cast<float>(get_parameter("force_band").as_double());
    force_target_ = static_cast<float>(get_parameter("force_target").as_double());
    close_rate_ = static_cast<float>(get_parameter("close_rate").as_double());
    trim_close_rate_ = static_cast<float>(get_parameter("trim_close_rate").as_double());
    release_rate_ = static_cast<float>(get_parameter("release_rate").as_double());
    lift_rate_ = static_cast<float>(get_parameter("lift_rate").as_double());
    stable_steps_required_ = get_parameter("stable_steps_required").as_int();
    max_probe_steps_ = get_parameter("max_probe_steps").as_int();
    close_steps_ = get_parameter("close_steps").as_int();
    lift_step_ = get_parameter("lift_step").as_int();
    reset_controller();
  }

  void reset_controller()
  {
    step_ = 0;
    stable_steps_ = 0;
    lifting_ = false;
    probe_stage_ = "approach";
    close_steps_left_ = 0;
    probe_steps_ = 0;
    probe_force_ = 0.0F;
    recovery_steps_left_ = 0;
  }

  void publish_command()
  {
    if (!builder_.ready()) {
      return;
    }

    const auto action = act();
    builder_.set_previous_action(action);
    auto msg = command_from_action(action);
    msg.header.stamp = now();
    command_pub_->publish(msg);
  }

  Action act()
  {
    switch (controller_type_) {
      case ControllerType::Fixed:
        return fixed_action();
      case ControllerType::Threshold:
        return threshold_action();
      case ControllerType::ProbeThenLift:
        return probe_then_lift_action();
      case ControllerType::SafeForce:
      default:
        return safe_force_action();
    }
  }

  Action fixed_action()
  {
    Action action{0.0F, 0.0F, 0.0F, 0.0F};
    if (step_ < close_steps_) {
      action = {0.0F, close_rate_, close_rate_, close_rate_};
    } else if (step_ >= lift_step_) {
      action = {lift_rate_, 0.0F, 0.0F, 0.0F};
    }
    ++step_;
    return clip_action(action);
  }

  Action threshold_action()
  {
    const auto forces = builder_.per_finger_max_taxel_force();
    const auto max_force = *std::max_element(forces.begin(), forces.end());
    lifting_ = lifting_ || max_force >= force_target_;
    ++step_;
    return lifting_ ?
           clip_action({lift_rate_, 0.0F, 0.0F, 0.0F}) :
           clip_action({0.0F, close_rate_, close_rate_, close_rate_});
  }

  Action safe_force_action()
  {
    const auto forces = builder_.per_finger_max_taxel_force();
    const auto low = target_force_ - force_band_;
    const auto high = target_force_ + force_band_;
    const auto contact_count = count_forces_at_least(forces, contact_threshold_);
    const auto ready_count = count_forces_at_least(forces, low);
    const auto max_force = *std::max_element(forces.begin(), forces.end());
    const bool balanced = ready_count >= 2 && max_force <= high;
    stable_steps_ = balanced ? stable_steps_ + 1 : 0;
    lifting_ = lifting_ || stable_steps_ >= stable_steps_required_ ||
      (step_ >= max_probe_steps_ && contact_count >= 2);

    Action action{0.0F, 0.0F, 0.0F, 0.0F};
    if (contact_count == 0) {
      action = {0.0F, close_rate_, close_rate_, close_rate_};
    } else {
      for (std::size_t i = 0; i < forces.size(); ++i) {
        action[i + 1] = finger_adjustment(forces[i], low, high);
      }
    }
    if (lifting_) {
      action[0] = lift_rate_;
      for (std::size_t i = 0; i < forces.size(); ++i) {
        action[i + 1] = finger_adjustment(forces[i], low * 0.85F, high);
      }
    }

    ++step_;
    return clip_action(action);
  }

  Action probe_then_lift_action()
  {
    const auto forces = builder_.per_finger_max_taxel_force();
    const auto max_force = *std::max_element(forces.begin(), forces.end());
    Action action{0.0F, 0.0F, 0.0F, 0.0F};

    if (probe_stage_ == "approach" && max_force > probe_contact_threshold_) {
      probe_stage_ = "grip";
      const auto profile = touch_profile(step_, max_force);
      active_close_rate_ = profile.first;
      close_steps_left_ = profile.second;
    }

    if (probe_stage_ == "approach") {
      action = {0.0F, close_rate_, close_rate_, close_rate_};
    } else if (probe_stage_ == "grip") {
      if (close_steps_left_ > 0) {
        action = {0.0F, active_close_rate_, active_close_rate_, active_close_rate_};
        --close_steps_left_;
      } else {
        probe_stage_ = "probe";
        probe_steps_ = 1;
        probe_force_ = max_force;
        action = {lift_rate_, 0.0F, 0.0F, 0.0F};
      }
    } else if (probe_stage_ == "probe") {
      if (probe_steps_ < micro_lift_steps_) {
        ++probe_steps_;
        action = {lift_rate_, 0.0F, 0.0F, 0.0F};
      } else if (max_force >= probe_force_ * retention_ratio_) {
        probe_stage_ = "lift";
        action = {lift_rate_, 0.0F, 0.0F, 0.0F};
      } else {
        probe_stage_ = "recover";
        recovery_steps_left_ = recovery_steps_;
        action = {-lift_rate_, 0.0F, 0.0F, 0.0F};
      }
    } else if (probe_stage_ == "recover") {
      if (recovery_steps_left_ > 1) {
        --recovery_steps_left_;
        action = {-lift_rate_, 0.0F, 0.0F, 0.0F};
      } else {
        probe_stage_ = "grip";
        active_close_rate_ = recovery_close_rate_;
        close_steps_left_ = 6;
        action = {-lift_rate_, 0.0F, 0.0F, 0.0F};
      }
    } else {
      action = {lift_rate_, 0.0F, 0.0F, 0.0F};
    }

    ++step_;
    return clip_action(action);
  }

  static int count_forces_at_least(const PerFingerForces & forces, float threshold)
  {
    return static_cast<int>(std::count_if(
      forces.begin(),
      forces.end(),
      [threshold](float force) { return force >= threshold; }));
  }

  float finger_adjustment(float force, float low, float high) const
  {
    if (force < low) {
      return trim_close_rate_;
    }
    if (force > high) {
      return -release_rate_;
    }
    return 0.0F;
  }

  static std::pair<float, int> touch_profile(int contact_step, float contact_force)
  {
    if (contact_force >= 0.15F) {
      return {0.05F, 18};
    }
    if (contact_step >= 77) {
      return {0.03F, 18};
    }
    if (contact_force >= 0.055F) {
      return {0.08F, 29};
    }
    return {0.05F, 21};
  }

  ObservationBuilder builder_;
  ControllerType controller_type_{ControllerType::SafeForce};
  rclcpp::Subscription<blindtouch_interfaces::msg::TactileState>::SharedPtr tactile_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<blindtouch_interfaces::msg::SimulationState>::SharedPtr sim_sub_;
  rclcpp::Publisher<blindtouch_interfaces::msg::ClawCommand>::SharedPtr command_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string last_outcome_;
  int step_{0};
  int stable_steps_{0};
  bool lifting_{false};
  float target_force_{0.50F};
  float force_band_{0.14F};
  float force_target_{0.65F};
  float close_rate_{0.18F};
  float trim_close_rate_{0.08F};
  float release_rate_{0.06F};
  float lift_rate_{1.0F};
  float contact_threshold_{0.035F};
  int stable_steps_required_{8};
  int max_probe_steps_{100};
  int close_steps_{24};
  int lift_step_{24};

  std::string probe_stage_{"approach"};
  float active_close_rate_{0.0F};
  int close_steps_left_{0};
  int probe_steps_{0};
  float probe_force_{0.0F};
  int recovery_steps_left_{0};
  int micro_lift_steps_{3};
  int recovery_steps_{3};
  float recovery_close_rate_{0.03F};
  float probe_contact_threshold_{1e-5F};
  float retention_ratio_{0.30F};
};

}  // namespace blindtouch_control

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<blindtouch_control::ScriptedControllerNode>());
  rclcpp::shutdown();
  return 0;
}

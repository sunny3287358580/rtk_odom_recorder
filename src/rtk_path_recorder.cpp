#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <bt982k6_msgs/msg/rtk_heading.hpp>

#include "geodetic.hpp"

class RtkOdomRecorder : public rclcpp::Node
{
public:
  RtkOdomRecorder() : Node("rtk_odom_recorder")
  {
    declare_parameter<std::string>("odom_topic", "/lio/robo/odom");
    declare_parameter<std::string>("rtk_fix_topic", "/rtk/fix");
    declare_parameter<std::string>("rtk_heading_topic", "/rtk/heading");
    declare_parameter<std::string>("save_dir", ".");
    declare_parameter<double>("save_interval_sec", 1.0);
    declare_parameter<int>("rtk_min_status", 2);
    declare_parameter<double>("origin.latitude", 0.0);
    declare_parameter<double>("origin.longitude", 0.0);
    declare_parameter<double>("origin.altitude", 0.0);
    declare_parameter<bool>("alignment.enabled", false);
    declare_parameter<double>("alignment.yaw_rad", 0.0);
    declare_parameter<double>("alignment.tx", 0.0);
    declare_parameter<double>("alignment.ty", 0.0);
    declare_parameter<double>("alignment.tz", 0.0);

    odom_topic_ = get_parameter("odom_topic").as_string();
    rtk_fix_topic_ = get_parameter("rtk_fix_topic").as_string();
    rtk_heading_topic_ = get_parameter("rtk_heading_topic").as_string();
    save_dir_ = get_parameter("save_dir").as_string();
    save_interval_sec_ = get_parameter("save_interval_sec").as_double();
    rtk_min_status_ = get_parameter("rtk_min_status").as_int();
    alignment_enabled_ = get_parameter("alignment.enabled").as_bool();
    alignment_yaw_ = get_parameter("alignment.yaw_rad").as_double();
    alignment_tx_ = get_parameter("alignment.tx").as_double();
    alignment_ty_ = get_parameter("alignment.ty").as_double();
    alignment_tz_ = get_parameter("alignment.tz").as_double();

    const double lat0 = get_parameter("origin.latitude").as_double();
    const double lon0 = get_parameter("origin.longitude").as_double();
    const double alt0 = get_parameter("origin.altitude").as_double();
    if (!std::isfinite(lat0) || !std::isfinite(lon0) || !std::isfinite(alt0) ||
        std::abs(lat0) > 90.0 || std::abs(lon0) > 180.0) {
      throw std::runtime_error("origin.{latitude,longitude,altitude} must be valid finite coordinates");
    }
    if (!(save_interval_sec_ > 0.0) || !std::isfinite(save_interval_sec_)) {
      throw std::runtime_error("save_interval_sec must be positive and finite");
    }
    if (alignment_enabled_ &&
        (!std::isfinite(alignment_yaw_) || !std::isfinite(alignment_tx_) ||
         !std::isfinite(alignment_ty_) || !std::isfinite(alignment_tz_))) {
      throw std::runtime_error("enabled alignment values must be finite");
    }

    enu_.setOrigin(lat0, lon0, alt0);
    std::filesystem::create_directories(save_dir_);
    const std::vector<std::string> output_names = {
      "lio_odom.csv", "rtk_fix.csv", "rtk_heading.csv"};
    for (const auto& name : output_names) {
      const auto path = std::filesystem::path(save_dir_) / name;
      if (std::filesystem::exists(path)) {
        throw std::runtime_error(
          "refusing to overwrite existing recording file: " + path.string());
      }
    }
    lio_file_.open(save_dir_ + "/lio_odom.csv", std::ios::out | std::ios::trunc);
    fix_file_.open(save_dir_ + "/rtk_fix.csv", std::ios::out | std::ios::trunc);
    heading_file_.open(save_dir_ + "/rtk_heading.csv", std::ios::out | std::ios::trunc);
    if (!lio_file_ || !fix_file_ || !heading_file_) {
      throw std::runtime_error("could not open one or more recorder output files in " + save_dir_);
    }
    lio_file_ << std::fixed << std::setprecision(9)
              << "timestamp,x,y,z,qx,qy,qz,qw\n";
    fix_file_ << std::fixed << std::setprecision(10)
              << "timestamp,latitude,longitude,altitude,status,east,north,up\n";
    heading_file_ << std::fixed << std::setprecision(10)
                  << "timestamp,valid,heading_rad,pitch_rad,baseline_length_m,"
                     "heading_stddev_rad,pitch_stddev_rad,satellites_tracked,"
                     "satellites_used,raw_solution_status,raw_position_type\n";

    auto qos = rclcpp::SensorDataQoS();
    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, qos,
      std::bind(&RtkOdomRecorder::odomCallback, this, std::placeholders::_1));
    sub_rtk_fix_ = create_subscription<sensor_msgs::msg::NavSatFix>(
      rtk_fix_topic_, qos,
      std::bind(&RtkOdomRecorder::rtkFixCallback, this, std::placeholders::_1));
    sub_rtk_heading_ = create_subscription<bt982k6_msgs::msg::RtkHeading>(
      rtk_heading_topic_, qos,
      std::bind(&RtkOdomRecorder::rtkHeadingCallback, this, std::placeholders::_1));

    path_enu_pub_ = create_publisher<nav_msgs::msg::Path>("/rtk/path_enu", 10);
    if (alignment_enabled_) {
      path_world_pub_ = create_publisher<nav_msgs::msg::Path>("/rtk/path_world", 10);
    }
    path_enu_.header.frame_id = "rtk_enu";
    path_world_.header.frame_id = "world";

    timer_ = create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(save_interval_sec_ * 1000.0)),
      std::bind(&RtkOdomRecorder::flushTimer, this));

    RCLCPP_INFO(get_logger(),
      "Recording to %s; odom=%s, fix=%s, heading=%s",
      save_dir_.c_str(), odom_topic_.c_str(),
      rtk_fix_topic_.c_str(), rtk_heading_topic_.c_str());
    RCLCPP_INFO(get_logger(),
      "ENU origin from parameters: lat=%.11f lon=%.11f alt=%.4f",
      lat0, lon0, alt0);
    if (alignment_enabled_) {
      RCLCPP_INFO(get_logger(),
        "RTK world alignment enabled: yaw=%.9f tx=%.4f ty=%.4f tz=%.4f",
        alignment_yaw_, alignment_tx_, alignment_ty_, alignment_tz_);
    } else {
      RCLCPP_WARN(get_logger(),
        "No RTK-to-world alignment loaded; publishing only /rtk/path_enu (frame rtk_enu)");
    }
  }

private:
  static double stampSeconds(const builtin_interfaces::msg::Time& stamp)
  {
    return static_cast<double>(stamp.sec) +
           static_cast<double>(stamp.nanosec) * 1e-9;
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& p = msg->pose.pose.position;
    const auto& q = msg->pose.pose.orientation;
    lio_file_ << stampSeconds(msg->header.stamp) << ','
              << p.x << ',' << p.y << ',' << p.z << ','
              << q.x << ',' << q.y << ',' << q.z << ',' << q.w << '\n';
    ++odom_count_;
  }

  void rtkFixCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
  {
    if (msg->status.status < rtk_min_status_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "RTK fix rejected by status filter (status=%d, required=%d)",
        static_cast<int>(msg->status.status), rtk_min_status_);
      return;
    }
    if (!std::isfinite(msg->latitude) || !std::isfinite(msg->longitude) ||
        !std::isfinite(msg->altitude)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "RTK fix contains non-finite coordinates; skipping");
      return;
    }

    double east, north, up;
    enu_.lla2enu(msg->latitude, msg->longitude, msg->altitude, east, north, up);
    std::lock_guard<std::mutex> lock(mutex_);
    const double t = stampSeconds(msg->header.stamp);
    fix_file_ << t << ',' << msg->latitude << ',' << msg->longitude << ','
              << msg->altitude << ',' << static_cast<int>(msg->status.status) << ','
              << east << ',' << north << ',' << up << '\n';

    geometry_msgs::msg::PoseStamped pose;
    pose.header = msg->header;
    pose.header.frame_id = "rtk_enu";
    pose.pose.position.x = east;
    pose.pose.position.y = north;
    pose.pose.position.z = up;
    pose.pose.orientation.w = 1.0;
    appendPath(path_enu_, pose, path_enu_pub_);

    if (alignment_enabled_) {
      const double c = std::cos(alignment_yaw_);
      const double s = std::sin(alignment_yaw_);
      pose.header.frame_id = "world";
      pose.pose.position.x = c * east - s * north + alignment_tx_;
      pose.pose.position.y = s * east + c * north + alignment_ty_;
      pose.pose.position.z = up + alignment_tz_;
      appendPath(path_world_, pose, path_world_pub_);
    }
    ++fix_count_;
  }

  void rtkHeadingCallback(const bt982k6_msgs::msg::RtkHeading::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    heading_file_ << stampSeconds(msg->header.stamp) << ','
                  << (msg->valid ? 1 : 0) << ','
                  << msg->heading_rad << ',' << msg->pitch_rad << ','
                  << msg->baseline_length_m << ',' << msg->heading_stddev_rad << ','
                  << msg->pitch_stddev_rad << ',' << msg->satellites_tracked << ','
                  << msg->satellites_used << ','
                  << std::quoted(msg->raw_solution_status) << ','
                  << std::quoted(msg->raw_position_type) << '\n';
    ++heading_count_;
  }

  void appendPath(
    nav_msgs::msg::Path& path,
    const geometry_msgs::msg::PoseStamped& pose,
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr& publisher)
  {
    path.poses.push_back(pose);
    path.header.stamp = pose.header.stamp;
    if (path.poses.size() > 200000) {
      path.poses.erase(path.poses.begin(), path.poses.begin() + 50000);
    }
    publisher->publish(path);
  }

  void flushTimer()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    lio_file_.flush();
    fix_file_.flush();
    heading_file_.flush();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
      "recorded lio=%zu, fixes=%zu, headings=%zu",
      odom_count_, fix_count_, heading_count_);
  }

  std::string odom_topic_;
  std::string rtk_fix_topic_;
  std::string rtk_heading_topic_;
  std::string save_dir_;
  double save_interval_sec_{1.0};
  int rtk_min_status_{2};
  bool alignment_enabled_{false};
  double alignment_yaw_{0.0};
  double alignment_tx_{0.0};
  double alignment_ty_{0.0};
  double alignment_tz_{0.0};

  std::ofstream lio_file_;
  std::ofstream fix_file_;
  std::ofstream heading_file_;
  std::mutex mutex_;
  geo::EnuFrame enu_;
  nav_msgs::msg::Path path_enu_;
  nav_msgs::msg::Path path_world_;
  size_t odom_count_{0};
  size_t fix_count_{0};
  size_t heading_count_{0};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_rtk_fix_;
  rclcpp::Subscription<bt982k6_msgs::msg::RtkHeading>::SharedPtr sub_rtk_heading_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_enu_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_world_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<RtkOdomRecorder>());
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("rtk_odom_recorder"), "%s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}

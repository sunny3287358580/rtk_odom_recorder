#pragma once
#include <cmath>
#include <Eigen/Dense>

namespace geo {

// WGS84
constexpr double kA  = 6378137.0;                  // 长半轴
constexpr double kF  = 1.0 / 298.257223563;        // 扁率
constexpr double kB  = kA * (1.0 - kF);            // 短半轴
constexpr double kE2 = kF * (2.0 - kF);            // 第一偏心率平方

inline void lla2ecef(double lat, double lon, double alt,
                     double& x, double& y, double& z)
{
  const double phi = lat * M_PI / 180.0;
  const double lam = lon * M_PI / 180.0;
  const double sp = std::sin(phi), cp = std::cos(phi);
  const double sl = std::sin(lam), cl = std::cos(lam);
  const double N  = kA / std::sqrt(1.0 - kE2 * sp * sp);
  x = (N + alt) * cp * cl;
  y = (N + alt) * cp * sl;
  z = (N * (1.0 - kE2) + alt) * sp;
}

// 由首点 (lat0,lon0,alt0) 构造 ENU 旋转矩阵，原点 ecef0
struct EnuFrame {
  Eigen::Vector3d ecef0;
  Eigen::Matrix3d R;   // ecef -> enu

  void setOrigin(double lat0, double lon0, double alt0) {
    lla2ecef(lat0, lon0, alt0, ecef0.x(), ecef0.y(), ecef0.z());
    const double phi = lat0 * M_PI / 180.0;
    const double lam = lon0 * M_PI / 180.0;
    const double sp = std::sin(phi), cp = std::cos(phi);
    const double sl = std::sin(lam), cl = std::cos(lam);
    // 每列是 E/N/U 在 ECEF 下的方向
    R << -sl,        cl,        0.0,
         -sp * cl,  -sp * sl,   cp,
          cp * cl,   cp * sl,   sp;
  }

  void lla2enu(double lat, double lon, double alt,
               double& e, double& n, double& u) const
  {
    double x, y, z;
    lla2ecef(lat, lon, alt, x, y, z);
    Eigen::Vector3d d(x - ecef0.x(), y - ecef0.y(), z - ecef0.z());
    Eigen::Vector3d enu = R * d;
    e = enu.x(); n = enu.y(); u = enu.z();
  }
};

}  // namespace geo
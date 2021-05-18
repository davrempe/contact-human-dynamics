

#ifndef GROUND_PLANE_H_
#define GROUND_PLANE_H_

#include <towr/terrain/height_map.h>

namespace towr {

/**
 * @brief Ground plane terrain with arbitrary orientation.
 */
class GroundPlane : public HeightMap {
public:
  using Ptr      = std::shared_ptr<GroundPlane>;
  GroundPlane();
  GroundPlane(Eigen::Vector3d normal, Eigen::Vector3d point);
  double GetHeight(double x, double y) const override;
  double GetHeightDerivWrtX(double x, double y) const override;
  double GetHeightDerivWrtY(double x, double y) const override;

  Eigen::Vector3d GetNormal() { return ground_normal_; }
  Eigen::Vector3d GetPoint() { return ground_point_; }

private:
  Eigen::Vector3d ground_normal_;
  Eigen::Vector3d ground_point_;
};

}

#endif /* GROUND_PLANE_H_ */
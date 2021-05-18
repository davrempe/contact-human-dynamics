

#include "terrain/ground_plane.h"

#include <iostream>

namespace towr {

GroundPlane::GroundPlane(){
    ground_normal_ << 0.0, 0.0, 1.0;
    ground_point_ << 0.0, 0.0, 0.0;
}

GroundPlane::GroundPlane(Eigen::Vector3d normal, Eigen::Vector3d point) : 
                            ground_normal_(normal), ground_point_(point) {
}

double
GroundPlane::GetHeight (double x, double y) const
{
  double z = -ground_normal_(1) * (y - ground_point_(1)) - ground_normal_(0) * (x - ground_point_(0));
  z /= ground_normal_(2);
  z += ground_point_(2);

//   std::cout << "(x, y, z) = (" << x << ", " << y << ", " << z << ")" << std::endl;
  return z;
}

double
GroundPlane::GetHeightDerivWrtX (double x, double y) const
{
  double dzdx = -ground_normal_(0) / ground_normal_(2);
  return dzdx;
}

double
GroundPlane::GetHeightDerivWrtY (double x, double y) const
{
  double dzdy = -ground_normal_(1) / ground_normal_(2);
  return dzdy;
}

}
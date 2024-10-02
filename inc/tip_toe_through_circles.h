#ifndef TIP_TOE_THROUGH_CIRCLES_H
#define TIP_TOE_THROUGH_CIRCLES_H

#include <vector>

/**
 * x, y- point coordinates.
 */
struct Point {
  double x, y;
  Point() : x(0.0), y(0.0) {}
  Point(double x, double y) : x(x), y(y) {}
};

/**
 * ctr - center Point.
 * r - radius
 */
struct Circle {
  Point ctr;
  double r;

  Circle() : ctr(), r(1.0) {}
  Circle(Point center, double radius) : ctr(center), r(radius) {}
  Circle(double center_x, double center_y, double radius)
      : ctr(center_x, center_y), r(radius)
  {
  }
};
//////////////////////////////////////////////////////////////

double shortest_path_length(const Point& a,
                            const Point& b,
                            const std::vector<Circle>& in_circles,
                            bool draw_system = true);

int run_all_test_cases(bool draw_system);
#endif

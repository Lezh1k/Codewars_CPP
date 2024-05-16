#include <assert.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>
#include <utility>
#include <vector>

static const double EPS = 1e-8;
#define DEBUG 1

#ifdef _UNIT_TESTS_
int main_tests(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
//////////////////////////////////////////////////////////////

struct Point {
  double x, y;

  Point() : x(0.0), y(0.0) {}
  Point(double x, double y) : x(x), y(y) {}
};

struct Circle {
  Point ctr;
  double r;

  Circle() : ctr(), r(1.0) {}
  Circle(Point c, double r) : ctr(c), r(r) {}
  Circle(double cx, double cy, double r) : ctr(cx, cy), r(r) {}
};

struct MyPoint {
  double x, y;

  MyPoint() : x(0.0), y(0.0) {}
  MyPoint(double x, double y) : x(x), y(y) {}
  friend bool operator<(const MyPoint& p1, const MyPoint& p2);
};

bool operator<(const MyPoint& p1, const MyPoint& p2)
{
  if (std::fabs(p1.x - p2.x) <= EPS) {
    return std::round(p1.y / EPS) * EPS < std::round(p2.y / EPS) * EPS;
  }
  return std::round(p1.x / EPS) * EPS < std::round(p2.x / EPS) * EPS;
}
inline bool operator==(const MyPoint& lhs, const MyPoint& rhs)
{
  return std::fabs(lhs.x - rhs.x) <= EPS && std::fabs(lhs.y - rhs.y) <= EPS;
}
inline bool operator!=(const MyPoint& lhs, const MyPoint& rhs)
{
  return !(lhs == rhs);
}
std::ostream& operator<<(std::ostream& os, const MyPoint& p)
{
  os << p.x << ":" << p.y;
  return os;
}
//////////////////////////////////////////////////////////////

struct MyCircle {
  MyPoint center;
  double radius;

  MyCircle() : center(), radius(1.0) {}
  MyCircle(MyPoint center, double radius) : center(center), radius(radius) {}
  MyCircle(double center_x, double center_y, double radius)
      : center(center_x, center_y), radius(radius)
  {
  }
};
//////////////////////////////////////////////////////////////

struct Line {
  double a, b, c;  // ax + by + c = 0

  Line() : a(0.), b(0.), c(0.) {}
  Line(double a, double b, double c) : a(a), b(b), c(c) {}

  void normalize()
  {
    double coeff = 1 / std::sqrt(a * a + b * b);
    coeff *= c > 0. ? -1 : 1;
    a *= coeff;
    b *= coeff;
    c *= coeff;
  }
};

struct LineSegment {
  MyPoint a, b;
  LineSegment() : a(), b() {}
  LineSegment(MyPoint a, MyPoint b) : a(a), b(b) {}
};

struct Arc {
  MyPoint a, b;   // points on circle
  double r;       // radius
  double thetha;  // angle between points

  Arc() : a(), b(), r(0.), thetha(0.) {}
  Arc(MyPoint a, MyPoint b, double r, double thetha)
      : a(a), b(b), r(r), thetha(thetha)
  {
  }
};
//////////////////////////////////////////////////////////////

struct Vertex {
  MyPoint point;
  double distance;

  Vertex() : point(), distance(0.) {}
  Vertex(const MyPoint& point, double distance)
      : point(point), distance(distance)
  {
  }
  friend bool operator<(const Vertex& v1, const Vertex& v2);
};

bool operator<(const Vertex& v1, const Vertex& v2)
{
  return v1.point < v2.point;
}
//////////////////////////////////////////////////////////////

typedef std::map<MyPoint, std::set<Vertex>> graph_t;
//////////////////////////////////////////////////////////////

static void DrawSystem(const std::set<MyPoint>& points,
                       const std::vector<MyCircle>& circles,
                       const std::vector<LineSegment>& lines,
                       const std::vector<MyPoint>& path);

static void tangents(const MyPoint& pt,
                     double r1,
                     double r2,
                     std::vector<Line>& dst);

static std::vector<Line> tangences(const MyCircle& c1, const MyCircle& c2);

/**
 * intersection_ntc: finds point of intersection of NORMALIZED TANGENT
 * line with the given circle. if line is not tangent to the given circle
 * assertion FAILS
 * */
static MyPoint intersection_ntc(const Line& l, const MyCircle& circle);

/**
 * intersection_lc: finds points of intersection of line and circle
 * */
static std::vector<MyPoint> intersection_lc(const Line& l,
                                            const MyCircle& circle);

/**
 * intersection_cc: finds points of intersection of two circles
 * */
static std::vector<MyPoint> intersection_cc(const MyCircle& c1,
                                            const MyCircle& c2);

static void associate_point_with_circle(
    const MyPoint& point,
    size_t c_idx,
    std::map<size_t, std::set<MyPoint>>& dct_circle_points);
static bool line_segment_cross_circle(const LineSegment& ls, const MyCircle& c);
static void neighbours(
    const MyCircle& c1,
    int c1_idx,
    const std::vector<MyCircle>& circles,
    graph_t& out_graph,
    std::vector<LineSegment>& out_line_segments,
    std::set<MyPoint>& out_points,
    std::map<size_t, std::set<MyPoint>>& dct_circle_points,
    const std::map<size_t, std::set<MyPoint>>& dct_circle_intersections);

static std::map<size_t, std::set<MyPoint>> find_all_circle_circle_intersections(
    const std::vector<MyCircle>& circles);
//////////////////////////////////////////////////////////////

void tangents(const MyPoint& pt, double r1, double r2, std::vector<Line>& dst)
{
  /*
   * System of equations with given x, y, r1 and r2:
   * a^2 + b^2 = 1;  // line normalization
   * c = r1
   * a*x + by + c = r2
   * */
  double r = r2 - r1;
  double z = (pt.x * pt.x) + (pt.y * pt.y);
  double d = z - r * r;
  if (d < -EPS) {
    return;
  }
  d = std::sqrt(std::fabs(d));
  double a = (pt.x * r + pt.y * d) / z;
  double b = (pt.y * r - pt.x * d) / z;
  double c = r1;

  Line l(a, b, c);
  dst.push_back(l);
}
//////////////////////////////////////////////////////////////

std::vector<Line> tangences(const MyCircle& c1, const MyCircle& c2)
{
  std::vector<Line> lines;
  for (int i = -1; i <= 1; i += 2) {
    for (int j = -1; j <= 1; j += 2) {
      MyPoint pt(c2.center.x - c1.center.x, c2.center.y - c1.center.y);
      tangents(pt, c1.radius * i, c2.radius * j, lines);
    }
  }

  // just move line
  for (auto& l : lines) {
    l.c -= l.a * c1.center.x + l.b * c1.center.y;
  }
  return lines;
}
//////////////////////////////////////////////////////////////

std::vector<MyPoint> intersection_lc(const Line& l, const MyCircle& circle)
{
  double a = l.a;
  double b = l.b;
  double c = l.c;
  double r = circle.radius;

  double x0 = -a * c / (a * a + b * b);
  double y0 = -b * c / (a * a + b * b);

  if (c * c > r * r * (a * a + b * b) + EPS) {
    return std::vector<MyPoint>();  // empty
  }

  if (std::fabs(c * c - r * r * (a * a + b * b)) < EPS) {
    // only tangece
    return std::vector<MyPoint>{MyPoint(x0, y0)};
  }

  double d = r * r - c * c / (a * a + b * b);
  double mult = std::sqrt(d / (a * a + b * b));
  double ax, ay, bx, by;
  ax = x0 + b * mult;
  bx = x0 - b * mult;
  ay = y0 - a * mult;
  by = y0 + a * mult;

  return std::vector<MyPoint>{MyPoint(ax, ay), MyPoint(bx, by)};
}
//////////////////////////////////////////////////////////////

std::vector<MyPoint> intersection_cc(const MyCircle& c1, const MyCircle& c2)
{
  // x^2 + y^2 = r1^2
  // (x - x2)^2 + (y - y2)^2 = r2^2
  // ====
  // x^2 + y^2 = r1^2
  // x*(-2*x2) + y*(-2*y2) + (x2^2 + y2^2 + r1^2 - r2^2) = 0
  // ====
  // Ax + Bx + C = 0
  // A = -2*x2
  // B = -2*y2
  // C = x2^2 + y2^2 + r1^2 - r2^2

  double x2 = c2.center.x - c1.center.x;
  double y2 = c2.center.y - c1.center.y;
  double r1 = c1.radius;
  double r2 = c2.radius;

  double a = -2 * x2;
  double b = -2 * y2;
  double c = x2 * x2 + y2 * y2 + r1 * r1 - r2 * r2;

  Line l(a, b, c);
  std::vector<MyPoint> ips = intersection_lc(l, c1);
  for (MyPoint& p : ips) {
    p.x += c1.center.x;
    p.y += c1.center.y;
  }
  return ips;
}
//////////////////////////////////////////////////////////////

MyPoint intersection_ntc(const Line& l, const MyCircle& circle)
{
  if (std::fabs(l.a * l.a + l.b * l.b - 1.) >= EPS) {
    throw std::invalid_argument("line is not normalized");
  }

  std::vector<MyPoint> res;
  double cx = circle.center.x;
  double cy = circle.center.y;
  double r = circle.radius;

  double a = l.a;
  double b = l.b;
  double c = l.c - (a * -cx + b * -cy);  // shift line

  double x0 = -a * c;
  double y0 = -b * c;

  double dcr = c * c - r * r;
  assert(std::fabs(dcr) <= EPS);  // they should be equal.

  return MyPoint(x0 + cx, y0 + cy);
}
//////////////////////////////////////////////////////////////

bool line_segment_cross_circle(const LineSegment& ls, const MyCircle& c)
{
  double x1 = ls.a.x;
  double y1 = ls.a.y;
  double x2 = ls.b.x;
  double y2 = ls.b.y;
  double x3 = c.center.x;
  double y3 = c.center.y;

  // check that neither point 1 nor point 2 are inside the circle
  double d1 = std::sqrt((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1));
  double d2 = std::sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
  if (d1 < c.radius || d2 < c.radius) {
    return true;
  }

  double u = (x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1);
  u /= (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);  // ||p2 - p1|| ^ 2

  if (u < 0. || u > 1.) {
    return false;  // point projection is out of line segment
  }

  double x = x1 + u * (x2 - x1);
  double y = y1 + u * (y2 - y1);

  double d = std::sqrt((x3 - x) * (x3 - x) + (y3 - y) * (y3 - y));
  return d < c.radius;
}
//////////////////////////////////////////////////////////////

void associate_point_with_circle(
    const MyPoint& point,
    size_t c_idx,
    std::map<size_t, std::set<MyPoint>>& dct_circle_points)
{
  std::map<size_t, std::set<MyPoint>>::iterator it =
      dct_circle_points.find(c_idx);
  if (it == dct_circle_points.end()) {
    auto ni = std::make_pair(c_idx, std::set<MyPoint>{point});
    dct_circle_points.insert(ni);
  } else {
    it->second.insert(point);
  }
}
//////////////////////////////////////////////////////////////

static double norm_atan2(double y, double x)
{
  double angle = atan2(y, x);
  if (angle < 0.)
    angle += 2 * M_PI;
  return angle;
}
//////////////////////////////////////////////////////////////

static void connect_points_on_circle(
    const MyCircle& c1,
    int c1_idx,
    std::map<size_t, std::set<MyPoint>>& dct_circle_points,
    const std::map<size_t, std::set<MyPoint>>& dct_circle_intersections,
    graph_t& out_graph)
{
  std::vector<MyPoint> circles_points;
  circles_points.reserve(dct_circle_points[c1_idx].size() +
                         dct_circle_intersections.at(c1_idx).size());
  for (MyPoint p : dct_circle_points[c1_idx]) {
    circles_points.push_back(p);
  }
  for (MyPoint p : dct_circle_intersections.at(c1_idx)) {
    circles_points.push_back(p);
  }

  std::sort(circles_points.begin(),
            circles_points.end(),
            [&c1](const MyPoint& p1, const MyPoint& p2) {
              double th1 = norm_atan2(p1.y - c1.center.y, p1.x - c1.center.x);
              double th2 = norm_atan2(p2.y - c1.center.y, p2.x - c1.center.x);
              return th1 < th2;
            });

  size_t si = 0;
  for (size_t i = 0; i < circles_points.size(); ++i) {
    si = circles_points.size() - 1 - i;
    if (dct_circle_intersections.at(c1_idx).find(circles_points[si]) !=
        dct_circle_intersections.at(c1_idx).end()) {
      break;
    }
  }

  for (size_t i = si; i < circles_points.size() + si; ++i) {
    size_t se = i + 1;
    for (; se < circles_points.size() + si; ++se) {
      if (dct_circle_intersections.at(c1_idx).find(
              circles_points[se % circles_points.size()]) !=
          dct_circle_intersections.at(c1_idx).end()) {
        break;
      }
    }

    size_t begin = i % circles_points.size();
    size_t end = se % circles_points.size();

    for (size_t ai = begin; ai % circles_points.size() != end; ++ai) {
      MyPoint a = circles_points[ai % circles_points.size()];
      for (size_t bi = ai; bi % circles_points.size() != end; ++bi) {
        MyPoint b = circles_points[bi % circles_points.size()];
        double ax = a.x - c1.center.x;
        double bx = b.x - c1.center.x;
        double ay = a.y - c1.center.y;
        double by = b.y - c1.center.y;

        double th_a = norm_atan2(ay, ax);
        double th_b = norm_atan2(by, bx);
        double d_th = th_b - th_a;
        if (d_th < 0.) {
          d_th += 2 * M_PI;
        }
        double d_ab = d_th * c1.radius;
        out_graph[a].insert(Vertex(b, d_ab));
        out_graph[b].insert(Vertex(a, d_ab));
      }  // for bi = ai+1; bi != end; ++bi
    }  // for ai = begin + 1; ai != end
  }  // for (size_t i = si; i < circles_points.size() + si; ++i)
}
//////////////////////////////////////////////////////////////

void neighbours(
    const MyCircle& c1,
    int c1_idx,
    const std::vector<MyCircle>& circles,
    graph_t& out_graph,
    std::vector<LineSegment>& out_line_segments,
    std::set<MyPoint>& out_points,
    std::map<size_t, std::set<MyPoint>>& dct_circle_points,
    const std::map<size_t, std::set<MyPoint>>& dct_circle_intersections)
{
  for (size_t c2_idx = c1_idx + 1; c2_idx < circles.size(); ++c2_idx) {
    const MyCircle& c2 = circles[c2_idx];
    std::vector<Line> tls = tangences(c1, c2);

    for (auto l : tls) {
      MyPoint pc1 = intersection_ntc(l, c1);
      MyPoint pc2 = intersection_ntc(l, c2);
      LineSegment ls(pc1, pc2);

      bool exclude = false;
      for (size_t k = 0; k < circles.size() && !exclude; ++k) {
        // we don't need to check src and dst circles
        if (k == static_cast<size_t>(c1_idx) || k == c2_idx)
          continue;
        exclude = line_segment_cross_circle(ls, circles[k]);
      }

      if (exclude)
        continue;

      double dx = pc1.x - pc2.x;
      double dy = pc1.y - pc2.y;
      double distance = std::sqrt(dx * dx + dy * dy);
      out_graph[pc1].insert(Vertex(pc2, distance));
      out_graph[pc2].insert(Vertex(pc1, distance));

      out_points.insert(pc1);
      out_points.insert(pc2);

      associate_point_with_circle(pc1, c1_idx, dct_circle_points);
      associate_point_with_circle(pc2, c2_idx, dct_circle_points);
      out_line_segments.push_back(ls);
    }  // for l : tls
  }  // for (j = i+1; j < circles.size(); ++j)

  // at this point all possible points are associated with circle C1.
  // need to connect them all with each other
  connect_points_on_circle(c1,
                           c1_idx,
                           dct_circle_points,
                           dct_circle_intersections,
                           out_graph);

}  // neigbours()
//////////////////////////////////////////////////////////////

std::map<size_t, std::set<MyPoint>> find_all_circle_circle_intersections(
    const std::vector<MyCircle>& circles)
{
  std::map<size_t, std::set<MyPoint>> dct_circle_intersections;
  for (size_t i = 0; i < circles.size(); ++i) {
    dct_circle_intersections.insert(std::make_pair(i, std::set<MyPoint>()));

    const MyCircle& c1 = circles[i];
    for (size_t j = 0; j < circles.size(); ++j) {
      if (i == j)
        continue;

      const MyCircle& c2 = circles[j];
      std::vector<MyPoint> cips = intersection_cc(c1, c2);
      for (const MyPoint& cip : cips) {
        associate_point_with_circle(cip, i, dct_circle_intersections);
        associate_point_with_circle(cip, j, dct_circle_intersections);
      }
    }
  }

  return dct_circle_intersections;
}
//////////////////////////////////////////////////////////////

double shortest_path_length(const Point& a,
                            const Point& b,
                            const std::vector<Circle>& in_circles)
{
  std::vector<MyCircle> circles = {MyCircle(a.x, a.y, 0.)};
  for (auto in_c : in_circles) {
    circles.push_back(MyCircle(in_c.ctr.x, in_c.ctr.y, in_c.r));
  }
  circles.push_back(MyCircle(b.x, b.y, 0.));

  std::vector<LineSegment> line_segments;
  std::set<MyPoint> points;

  // key - index of circle
  // value - set of points on this circle
  std::map<size_t, std::set<MyPoint>> dct_circle_points;
  // points where circle intersects with other circles
  std::map<size_t, std::set<MyPoint>> dct_circle_intersections =
      find_all_circle_circle_intersections(circles);

  graph_t graph;
  for (size_t i = 0; i < circles.size(); ++i) {
    neighbours(circles[i],
               i,
               circles,
               graph,
               line_segments,
               points,
               dct_circle_points,
               dct_circle_intersections);
  }

#if DEBUG
  std::cout << "\n\tGRAPH:\n\n";
  for (auto it : graph) {
    std::cout << it.first << "\n";
    for (auto v : it.second) {
      std::cout << "\t" << v.point << " " << v.distance << "\n";
    }
  }
  std::cout << "\n\t\\GRAPH\n";
#endif

  // find shortest path (Dijkstra)
  static const double INF = 10000000.;
  static const MyPoint IMPOSSIBLE_POINT(std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max());
  std::map<MyPoint, double> d;
  std::map<MyPoint, bool> u;
  std::map<MyPoint, MyPoint> p;

  // init D, U and P
  MyPoint a_ = MyPoint(a.x, a.y);
  MyPoint b_ = MyPoint(b.x, b.y);
  if (graph.find(b_) == graph.end()) {
    return -1.;
  }

  for (const auto& node : graph) {
    d.insert(std::make_pair(MyPoint(node.first), INF));
    u.insert(std::make_pair(MyPoint(node.first), false));
    p.insert(std::make_pair(MyPoint(node.first), IMPOSSIBLE_POINT));
  }
  d[a_] = 0.;

  for (size_t i = 0; i < graph.size(); ++i) {
    MyPoint v;
    bool first = true;
    for (const auto& j : graph) {
      if (u[j.first])
        continue;

      if (first || d[j.first] < d[v]) {
        v = j.first;
        first = false;
      }
    }  // for j : graph

    if (d[v] == INF)
      break;

    u[v] = true;
    for (const auto& j : graph.at(v)) {
      MyPoint to = j.point;
      if (d[v] + j.distance < d[to]) {
        d[to] = d[v] + j.distance;
        p[to] = v;
      }
    }
  }  // end of Dijkstra's alg

#if DEBUG
  std::vector<MyPoint> path;
  for (MyPoint v = b_; v != a_ && v != IMPOSSIBLE_POINT; v = p[v]) {
    path.push_back(v);
  }
  path.push_back(a_);
  std::reverse(path.begin(), path.end());

  std::cout << "\nPATH:\n";
  for (auto p : path) {
    std::cout << p << "  " << d[p] << "\n";
  }
  DrawSystem(points, circles, line_segments, path);
#endif
  return d[b_] == INF ? -1. : d[b_];
}
//////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
#ifdef _UNIT_TESTS_
  return main_tests(argc, argv);
#else
  (void)argc;
  (void)argv;

  struct test_case_t {
    Point a, b;
    std::vector<Circle> c;
    double expected;
  };

  test_case_t t1 = {
      .a = {-3, 1},
      .b = {4.25, 0},
      .c = {{0.0, 0.0, 2.5},
            {1.5, 2.0, 0.5},
            {3.5, 1.0, 1.0},
            {3.5, -1.7, 1.2}},
      .expected = 9.11821650244,
  };
  test_case_t t2 = {
      .a = {0, 1},
      .b = {0, -1},
      .c = {{0.0, 0.0, 0.8},
            {3.8, 0.0, 3.2},
            {-3.5, 0.0, 3.0},
            {-7.0, 0.0, 1.0}},
      .expected = 19.0575347577,
  };

  test_case_t t3 = {
      .a = {3, 0},
      .b = {0, 4},
      .c = {{0, 0, 1}},
      .expected = 5.,
  };

  test_case_t t4 = {
      .a = {0, 0},
      .b = {20, 20},
      .c = {{4, 0, 3}, {-4, 0, 3}, {0, 4, 3}, {0, -4, 3}},
      .expected = -1.,
  };

  test_case_t t5 = {
      .a = {0, 1},
      .b = {0, -1},
      .c = {{0.0, 0.0, 0.8},
            {-3.8, 0.0, 3.2},
            {3.5, 0.0, 3.0},
            {7.0, 0.0, 1.0}},
      .expected = 19.0575347577,
  };

  test_case_t t6 = {
      .a = {0, -7},
      .b = {8, 8},
      .c = {},
      .expected = 17.0,
  };

  test_case_t t7 = {
      .a = {0.5, 0.5},
      .b = {2, 2},
      .c = {{0, 0, 1}},
      .expected = -1.,
  };

  test_case_t t8 = {
      .a = {2, 2},
      .b = {0.5, 0.5},
      .c = {{0, 0, 1}},
      .expected = -1.,
  };

  test_case_t t9 = {
      .a = {-2.51672, 1.30616},
      .b = {3.87272, -1.91084},
      .c =
          {
            {-0.597119, 0.88165, 0.91446},  {-0.962345, -4.03892, 1.30996},
            {-4.39527, 0.598278, 1.14107},  {-4.04069, -3.12178, 1.03569},
            {-4.75432, -4.2065, 0.668657},  {-0.035397, -0.376569, 1.08104},
            {-3.09102, 3.95867, 1.3898},    {2.99264, 1.72149, 0.850079},
            {3.63657, 0.886315, 1.25232},   {3.49616, 2.21439, 1.23538},
            {1.00417, -2.31594, 0.470425},  {0.848565, -2.36409, 0.873431},
            {-1.33173, -1.14513, 1.33513},  {2.36933, 4.22031, 1.2208},
            {2.87691, 1.33573, 0.733986},   {2.83323, -1.17842, 0.67553},
            {-2.05985, 3.95228, 0.900663},  {0.601596, -1.76171, 0.99692},
            {0.378449, 1.05643, 0.667554},  {-4.80266, 3.96277, 0.798043},
            {0.228887, 3.70402, 1.23521},   {-4.57708, 0.916904, 1.20442},
            {-3.53017, 1.16296, 0.842481},  {-1.25862, 3.24817, 0.927703},
            {-3.52648, -1.43026, 0.508443}, {-4.64439, -3.50993, 1.43188},
            {1.5644, -3.07672, 0.909408},   {-2.02056, 2.09043, 0.909134},
            {0.897352, 3.06997, 0.938222},  {4.44652, -3.77101, 1.07068},
            {2.72584, 3.64219, 1.05714},    {-1.20085, -1.2677, 0.33171},
            {4.45392, 1.81521, 0.78651},    {1.08901, -1.50047, 0.767389},
            {-0.5909, 2.99315, 0.957265},   {0.555153, -2.07293, 1.1748},
            {-2.21517, -1.12519, 0.783175}, {0.431842, 2.53361, 0.766385},
            },
      .expected = 8.26474,
  };

  test_case_t tA = {
      .a = {2, 0},
      .b = {-3, 3},
      .c = {{0, 0, 2}},
      .expected = 6.29421907015,
  };

  test_case_t tB = {
      .a = {3, 5},
      .b = {3, -5},
      .c =
          {
            {0, 0, 3},
            {5, 0, 2},
            },
      .expected = 10.,
  };

  test_case_t* test_cases[] = {
      /* &t1, */
      /* &t2, */
      /* &t3, */
      /* &t4, */
      /* &t5, */
      /* &t6, */
      /* &t7, */
      /* &t8, */
      &t9,
      /* &tA, */
      /* &tB, */
      nullptr,
  };

  for (test_case_t** tc = test_cases; *tc; ++tc) {
    try {
      test_case_t* ptc = *tc;
      double act = shortest_path_length(ptc->a, ptc->b, ptc->c);
      std::cout << std::setprecision(12) << act << " == " << ptc->expected;
      std::cout << "\t" << (std::fabs(act - ptc->expected) <= EPS) << "\n";
    } catch (const std::exception& exc) {
      std::cout << exc.what() << std::endl;
      return 1;
    }
  }

  return 0;
#endif
}
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

void DrawSystem(const std::set<MyPoint>& points,
                const std::vector<MyCircle>& circles,
                const std::vector<LineSegment>& lines,
                const std::vector<MyPoint>& path)
{
  const char* wn = "TipToe through the circle";
  const double scale = 90.;
  const int width = 1400;
  const int height = 1000;
  const int center_offset_x = width / 2;
  const int center_offset_y = height / 2;
  cv::Mat mat_img(height, width, CV_8UC4, cv::Scalar(255, 255, 255));

  cv::Scalar circle_color(0xe0, 0xe0, 0xe0);
  cv::Scalar point_color(0, 0, 0);
  cv::Scalar line_color(0x00, 0xff, 0x00);
  cv::Scalar path_color(0x00, 0x00, 0xff);
  cv::Scalar start_color(0xff, 0x00, 0x00);
  cv::Scalar end_color(0xff, 0xff, 0x00);

  cv::namedWindow(wn);

  for (auto& c : circles) {
    cv::circle(mat_img,
               cv::Point(c.center.x * scale + center_offset_x,
                         c.center.y * scale + center_offset_y),
               c.radius * scale,
               circle_color);
  }

  cv::line(mat_img,
           cv::Point(width / 2, 0),
           cv::Point(width / 2, height),
           point_color);
  cv::line(mat_img,
           cv::Point(0, height / 2),
           cv::Point(width, height / 2),
           point_color);

  for (auto& ls : lines) {
    cv::line(mat_img,
             cv::Point(ls.a.x * scale + center_offset_x,
                       ls.a.y * scale + center_offset_y),
             cv::Point(ls.b.x * scale + center_offset_x,
                       ls.b.y * scale + center_offset_y),
             line_color);
  }

  for (auto& p : points) {
    cv::circle(
        mat_img,
        cv::Point(p.x * scale + center_offset_x, p.y * scale + center_offset_y),
        1,
        point_color,
        -1);
  }

  for (size_t i = 0; i < path.size() - 1; ++i) {
    cv::Point a(path[i].x * scale + center_offset_x,
                path[i].y * scale + center_offset_y);
    cv::Point b(path[i + 1].x * scale + center_offset_x,
                path[i + 1].y * scale + center_offset_y);
    cv::line(mat_img, a, b, path_color);
  }

  cv::circle(mat_img,
             cv::Point(path.front().x * scale + center_offset_x,
                       path.front().y * scale + center_offset_y),
             1,
             start_color,
             -1);
  cv::circle(mat_img,
             cv::Point(path.back().x * scale + center_offset_x,
                       path.back().y * scale + center_offset_y),
             1,
             end_color,
             -1);

  cv::flip(mat_img, mat_img, 0);
  cv::imshow(wn, mat_img);
  while (cv::waitKey(1) != 0x1b)  // ESC
    ;
}
//////////////////////////////////////////////////////////////

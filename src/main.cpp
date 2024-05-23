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

static const double EPS = 1e-9;
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

bool operator<(const Point& p1, const Point& p2)
{
  if (std::fabs(p1.x - p2.x) <= EPS) {
    return std::round(p1.y / EPS) * EPS < std::round(p2.y / EPS) * EPS;
  }
  return std::round(p1.x / EPS) * EPS < std::round(p2.x / EPS) * EPS;
}
inline bool operator==(const Point& lhs, const Point& rhs)
{
  return std::fabs(lhs.x - rhs.x) <= EPS && std::fabs(lhs.y - rhs.y) <= EPS;
}
inline bool operator!=(const Point& lhs, const Point& rhs)
{
  return !(lhs == rhs);
}
std::ostream& operator<<(std::ostream& os, const Point& p)
{
  os << p.x << ":" << p.y;
  return os;
}
//////////////////////////////////////////////////////////////

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
  Point a, b;
  LineSegment() : a(), b() {}
  LineSegment(Point a, Point b) : a(a), b(b) {}
};

struct Arc {
  Point a, b;     // points on circle
  double r;       // radius
  double thetha;  // angle between points

  Arc() : a(), b(), r(0.), thetha(0.) {}
  Arc(Point a, Point b, double r, double thetha)
      : a(a), b(b), r(r), thetha(thetha)
  {
  }
};
//////////////////////////////////////////////////////////////

struct Vertex {
  Point point;
  double distance;

  Vertex() : point(), distance(0.) {}
  Vertex(const Point& point, double distance) : point(point), distance(distance)
  {
  }
  friend bool operator<(const Vertex& v1, const Vertex& v2);
};

bool operator<(const Vertex& v1, const Vertex& v2)
{
  return v1.point < v2.point;
}
//////////////////////////////////////////////////////////////

typedef std::map<Point, std::set<Vertex>> graph_t;
//////////////////////////////////////////////////////////////

static void DrawSystem(
    const std::set<Point>& points,
    const std::vector<Circle>& circles,
    const std::vector<LineSegment>& lines,
    const std::vector<Point>& path,
    const std::map<size_t, std::set<Point>>& dct_circle_intersections);

static void tangents(const Point& pt,
                     double r1,
                     double r2,
                     std::vector<Line>& dst);

static std::vector<Line> tangences(const Circle& c1, const Circle& c2);

/**
 * intersection_ntc: finds point of intersection of NORMALIZED TANGENT
 * line with the given circle. if line is not tangent to the given circle
 * assertion FAILS
 * */
static Point intersection_ntc(const Line& l, const Circle& circle);

/**
 * intersection_lc: finds points of intersection of line and circle
 * */
static std::vector<Point> intersection_lc(const Line& l, const Circle& circle);

/**
 * intersection_cc: finds points of intersection of two circles
 * */
static std::vector<Point> intersection_cc(const Circle& c1, const Circle& c2);

static void associate_point_with_circle(
    const Point& point,
    size_t c_idx,
    std::map<size_t, std::set<Point>>& dct_circle_points);
static bool line_segment_cross_circle(const LineSegment& ls, const Circle& c);
static void neighbours(
    const Circle& c1,
    int c1_idx,
    const std::vector<Circle>& circles,
    graph_t& out_graph,
    std::vector<LineSegment>& out_line_segments,
    std::set<Point>& out_points,
    std::map<size_t, std::set<Point>>& dct_circle_points,
    const std::map<size_t, std::set<Point>>& dct_circle_intersections);

static std::map<size_t, std::set<Point>> find_all_circle_circle_intersections(
    const std::vector<Circle>& circles);
//////////////////////////////////////////////////////////////

void tangents(const Point& pt, double r1, double r2, std::vector<Line>& dst)
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

std::vector<Line> tangences(const Circle& c1, const Circle& c2)
{
  std::vector<Line> lines;
  for (int i = -1; i <= 1; i += 2) {
    for (int j = -1; j <= 1; j += 2) {
      Point pt(c2.ctr.x - c1.ctr.x, c2.ctr.y - c1.ctr.y);
      tangents(pt, c1.r * i, c2.r * j, lines);
    }
  }

  // just move line
  for (auto& l : lines) {
    l.c -= l.a * c1.ctr.x + l.b * c1.ctr.y;
  }
  return lines;
}
//////////////////////////////////////////////////////////////

std::vector<Point> intersection_lc(const Line& l, const Circle& circle)
{
  double a = l.a;
  double b = l.b;
  double c = l.c;
  double r = circle.r;

  double x0 = -a * c / (a * a + b * b);
  double y0 = -b * c / (a * a + b * b);

  if (c * c > r * r * (a * a + b * b) + EPS) {
    return std::vector<Point>();  // empty
  }

  if (std::fabs(c * c - r * r * (a * a + b * b)) < EPS) {
    // only tangece
    return std::vector<Point>{Point(x0, y0)};
  }

  double d = r * r - c * c / (a * a + b * b);
  double mult = std::sqrt(d / (a * a + b * b));
  double ax, ay, bx, by;
  ax = x0 + b * mult;
  bx = x0 - b * mult;
  ay = y0 - a * mult;
  by = y0 + a * mult;

  return std::vector<Point>{Point(ax, ay), Point(bx, by)};
}
//////////////////////////////////////////////////////////////

std::vector<Point> intersection_cc(const Circle& c1, const Circle& c2)
{
  double dx = (c2.ctr.x - c1.ctr.x) * (c2.ctr.x - c1.ctr.x);
  double dy = (c2.ctr.y - c1.ctr.y) * (c2.ctr.y - c1.ctr.y);
  if (std::sqrt(dx + dy) > (c1.r + c2.r)) {
    return std::vector<Point>();
  }
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

  double x2 = c2.ctr.x - c1.ctr.x;
  double y2 = c2.ctr.y - c1.ctr.y;
  double r1 = c1.r;
  double r2 = c2.r;

  double a = -2 * x2;
  double b = -2 * y2;
  double c = x2 * x2 + y2 * y2 + r1 * r1 - r2 * r2;

  Line l(a, b, c);
  std::vector<Point> ips = intersection_lc(l, c1);
  for (Point& p : ips) {
    p.x += c1.ctr.x;
    p.y += c1.ctr.y;
  }
  return ips;
}
//////////////////////////////////////////////////////////////

Point intersection_ntc(const Line& l, const Circle& circle)
{
  if (std::fabs(l.a * l.a + l.b * l.b - 1.) >= EPS) {
    throw std::invalid_argument("line is not normalized");
  }

  std::vector<Point> res;
  double cx = circle.ctr.x;
  double cy = circle.ctr.y;
  double r = circle.r;

  double a = l.a;
  double b = l.b;
  double c = l.c - (a * -cx + b * -cy);  // shift line

  double x0 = -a * c;
  double y0 = -b * c;

  double dcr = c * c - r * r;
  assert(std::fabs(dcr) <= EPS);  // they should be equal.

  return Point(x0 + cx, y0 + cy);
}
//////////////////////////////////////////////////////////////

bool line_segment_cross_circle(const LineSegment& ls, const Circle& c)
{
  double x1 = ls.a.x;
  double y1 = ls.a.y;
  double x2 = ls.b.x;
  double y2 = ls.b.y;
  double x3 = c.ctr.x;
  double y3 = c.ctr.y;

  // check that neither point 1 nor point 2 are inside the circle
  double d1 = std::sqrt((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1));
  double d2 = std::sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
  if (d1 < c.r || d2 < c.r) {
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
  return d < c.r;
}
//////////////////////////////////////////////////////////////

void associate_point_with_circle(
    const Point& point,
    size_t c_idx,
    std::map<size_t, std::set<Point>>& dct_circle_points)
{
  std::map<size_t, std::set<Point>>::iterator it =
      dct_circle_points.find(c_idx);
  if (it == dct_circle_points.end()) {
    auto ni = std::make_pair(c_idx, std::set<Point>{point});
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

static void connect_points_on_circle_no_intersections(
    const Circle& c1,
    const std::vector<Point>& points,
    graph_t& out_graph)
{
  for (size_t i = 0; i < points.size(); ++i) {
    const Point& a = points[i];
    double ax = a.x - c1.ctr.x;
    double ay = a.y - c1.ctr.y;

    for (size_t j = i + 1; j < points.size(); ++j) {
      const Point& b = points[j];
      double bx = b.x - c1.ctr.x;
      double by = b.y - c1.ctr.y;

      double cos_th = ax * bx + ay * by;
      cos_th /= c1.r * c1.r;

      double th = std::acos(cos_th);
      double d_ab = th * c1.r;

      out_graph[a].insert(Vertex(b, d_ab));
      out_graph[b].insert(Vertex(a, d_ab));
    }  // for j
  }  // for i
}
//////////////////////////////////////////////////////////////

static void connect_points_on_circle(
    const Circle& c1,
    int c1_idx,
    std::map<size_t, std::set<Point>>& dct_circle_points,
    const std::map<size_t, std::set<Point>>& dct_circle_intersections,
    graph_t& out_graph)
{
  std::vector<Point> circles_points;
  circles_points.reserve(dct_circle_points[c1_idx].size() +
                         dct_circle_intersections.at(c1_idx).size());
  for (Point p : dct_circle_points[c1_idx]) {
    circles_points.push_back(p);
  }
  if (dct_circle_intersections.at(c1_idx).empty()) {
    connect_points_on_circle_no_intersections(c1, circles_points, out_graph);
    return;
  }

  for (Point p : dct_circle_intersections.at(c1_idx)) {
    circles_points.push_back(p);
  }

  std::sort(circles_points.begin(),
            circles_points.end(),
            [&c1](const Point& p1, const Point& p2) {
              double th1 = norm_atan2(p1.y - c1.ctr.y, p1.x - c1.ctr.x);
              double th2 = norm_atan2(p2.y - c1.ctr.y, p2.x - c1.ctr.x);
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
    if (begin == end) {
      begin = 0;
      end = circles_points.size() - 1;
    }

    for (size_t ai = begin; ai % circles_points.size() != end; ++ai) {
      Point a = circles_points[ai % circles_points.size()];
      double ax = a.x - c1.ctr.x;
      double ay = a.y - c1.ctr.y;
      double th_a = norm_atan2(ay, ax);

      for (size_t bi = ai + 1; bi % circles_points.size() != end; ++bi) {
        Point b = circles_points[bi % circles_points.size()];
        double bx = b.x - c1.ctr.x;
        double by = b.y - c1.ctr.y;
        double th_b = norm_atan2(by, bx);

        double d_th = th_b - th_a;
        if (d_th < 0.) {
          d_th += 2 * M_PI;
        }
        double d_ab = d_th * c1.r;

        out_graph[a].insert(Vertex(b, d_ab));
        out_graph[b].insert(Vertex(a, d_ab));
      }  // for bi = ai+1; bi != end; ++bi
    }  // for ai = begin; ai != end
  }  // for (size_t i = si; i < circles_points.size() + si; ++i)
}
//////////////////////////////////////////////////////////////

void neighbours(
    const Circle& c1,
    int c1_idx,
    const std::vector<Circle>& circles,
    graph_t& out_graph,
    std::vector<LineSegment>& out_line_segments,
    std::set<Point>& out_points,
    std::map<size_t, std::set<Point>>& dct_circle_points,
    const std::map<size_t, std::set<Point>>& dct_circle_intersections)
{
  for (size_t c2_idx = c1_idx + 1; c2_idx < circles.size(); ++c2_idx) {
    const Circle& c2 = circles[c2_idx];
    std::vector<Line> tls = tangences(c1, c2);

    for (auto l : tls) {
      Point pc1 = intersection_ntc(l, c1);
      Point pc2 = intersection_ntc(l, c2);
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

std::map<size_t, std::set<Point>> find_all_circle_circle_intersections(
    const std::vector<Circle>& circles)
{
  std::map<size_t, std::set<Point>> dct_circle_intersections;
  for (size_t i = 0; i < circles.size(); ++i) {
    dct_circle_intersections.insert(std::make_pair(i, std::set<Point>()));

    const Circle& c1 = circles[i];
    for (size_t j = 0; j < circles.size(); ++j) {
      if (i == j)
        continue;

      const Circle& c2 = circles[j];
      std::vector<Point> cips = intersection_cc(c1, c2);
      for (const Point& cip : cips) {
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
  std::vector<Circle> circles = {Circle(a.x, a.y, 0.)};
  for (auto in_c : in_circles) {
    circles.push_back(Circle(in_c.ctr.x, in_c.ctr.y, in_c.r));
  }
  circles.push_back(Circle(b.x, b.y, 0.));

  std::vector<LineSegment> line_segments;
  std::set<Point> points;

  // key - index of circle
  // value - set of points on this circle
  std::map<size_t, std::set<Point>> dct_circle_points;
  // points where circle intersects with other circles
  std::map<size_t, std::set<Point>> dct_circle_intersections =
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

  // find shortest path (Dijkstra)
  static const double INF = 10000000.;
  static const Point IMPOSSIBLE_POINT(std::numeric_limits<double>::max(),
                                      std::numeric_limits<double>::max());
  std::map<Point, double> d;
  std::map<Point, bool> u;
  std::map<Point, Point> p;

  // init D, U and P
  Point a_ = Point(a.x, a.y);
  Point b_ = Point(b.x, b.y);
  if (graph.find(b_) == graph.end()) {
    return -1.;
  }

  for (const auto& node : graph) {
    d.insert(std::make_pair(Point(node.first), INF));
    u.insert(std::make_pair(Point(node.first), false));
    p.insert(std::make_pair(Point(node.first), IMPOSSIBLE_POINT));
  }
  d[a_] = 0.;

  for (size_t i = 0; i < graph.size(); ++i) {
    Point v;
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
      Point to = j.point;
      if (d[v] + j.distance < d[to]) {
        d[to] = d[v] + j.distance;
        p[to] = v;
      }
    }
  }  // end of Dijkstra's alg

#if DEBUG
  std::vector<Point> path;
  for (Point v = b_; v != a_ && v != IMPOSSIBLE_POINT; v = p[v]) {
    path.push_back(v);
  }
  path.push_back(a_);
  std::reverse(path.begin(), path.end());
  DrawSystem(points, circles, line_segments, path, dct_circle_intersections);
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
      .a = {-2.51671803324, 1.30616030283},
      .b = {3.87271832814, -1.91084126197},
      .c =
          {
            {-0.59711858863, 0.881649872754, 0.914459906914},
            {-0.962345076259, -4.03892196016, 1.30995611332},
            {-4.39527013572, 0.598277573008, 1.14106745452},
            {-4.04069135198, -3.12177702552, 1.03569137848},
            {-4.75431525847, -4.20650289627, 0.66865736011},
            {-0.0353969936259, -0.376568927895, 1.08103569236},
            {-3.09102221159, 3.9586655586, 1.38980286361},
            {2.99263592577, 1.72149120132, 0.850078982138},
            {3.63656748319, 0.886315449607, 1.25231607005},
            {3.49615606247, 2.21439161571, 1.2353831067},
            {1.00416972535, -2.31594216311, 0.470424721367},
            {0.848565415945, -2.36408579862, 0.873430584953},
            {-1.33173195412, -1.14513431909, 1.33512805726},
            {2.36933284206, 4.22030604677, 1.22080219288},
            {2.87690920057, 1.33573198458, 0.733985557198},
            {2.83323324984, -1.17841666332, 0.675529731461},
            {-2.05984547501, 3.95228252513, 0.900663372944},
            {0.601596075576, -1.76170821069, 0.996919863974},
            {0.378448769916, 1.05643301969, 0.66755388116},
            {-4.80265962193, 3.96276660496, 0.798043358536},
            {0.228886886034, 3.70402431814, 1.23521347332},
            {-4.57708235132, 0.916903645266, 1.20442000937},
            {-3.53017202346, 1.16296296706, 0.842480887636},
            {-1.25862282002, 3.24817231158, 0.927702947729},
            {-3.52647561347, -1.43026176142, 0.508443075116},
            {-4.64438741328, -3.50993065862, 1.43187699087},
            {1.56440264778, -3.07671953226, 0.909407779551},
            {-2.02055817703, 2.09043149138, 0.909134115116},
            {0.897351747844, 3.06996929226, 0.93822169851},
            {4.44651678903, -3.77100617392, 1.07067507629},
            {2.72584294202, 3.64219413837, 1.0571402801},
            {-1.20084762806, -1.26770091942, 0.331709638634},
            {4.4539240771, 1.8152130465, 0.78650989763},
            {1.08900851337, -1.50046768365, 0.76738881052},
            {-0.590899854433, 2.99314968055, 0.957265331945},
            {0.555152648594, -2.07292616135, 1.17480092293},
            {-2.21516946098, -1.12518788548, 0.783175178501},
            {0.43184221955, 2.53361074021, 0.766384983971},
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

  test_case_t t76 = {
      .a = {1, 1},
      .b = {9, 9},
      .c =
          {
            {0, 0, 0.64115},   {0, 1, 0.132413},   {0, 2, 0.360349},
            {0, 3, 0.324987},  {0, 4, 0.291204},   {0, 5, 0.482743},
            {0, 6, 0.357549},  {0, 7, 0.472708},   {0, 8, 0.487758},
            {0, 9, 0.502299},  {0, 10, 0.291764},  {1, 0, 0.253298},
            {1, 2, 0.289496},  {1, 3, 0.487209},   {1, 4, 0.205027},
            {1, 5, 0.710598},  {1, 6, 0.355356},   {1, 7, 0.474757},
            {1, 8, 0.15398},   {1, 9, 0.552585},   {1, 10, 0.354647},
            {2, 0, 0.374451},  {2, 1, 0.417562},   {2, 2, 0.80148},
            {2, 3, 0.226192},  {2, 4, 0.256702},   {2, 5, 0.355266},
            {2, 6, 0.409288},  {2, 7, 0.327123},   {2, 8, 0.302255},
            {2, 9, 0.331616},  {2, 10, 0.116894},  {3, 0, 0.461943},
            {3, 1, 0.665481},  {3, 2, 0.472481},   {3, 3, 0.184833},
            {3, 4, 0.332489},  {3, 5, 0.454663},   {3, 6, 0.368843},
            {3, 7, 0.273874},  {3, 8, 0.499421},   {3, 9, 0.282398},
            {3, 10, 0.393656}, {4, 0, 0.303524},   {4, 1, 0.395856},
            {4, 2, 0.689242},  {4, 3, 0.347418},   {4, 4, 0.366592},
            {4, 5, 0.283776},  {4, 6, 0.293723},   {4, 7, 0.598069},
            {4, 8, 0.327934},  {4, 9, 0.693837},   {4, 10, 0.371845},
            {5, 0, 0.530865},  {5, 1, 0.485497},   {5, 2, 0.539198},
            {5, 3, 0.21921},   {5, 4, 0.373822},   {5, 5, 0.621798},
            {5, 6, 0.344001},  {5, 7, 0.498881},   {5, 8, 0.314385},
            {5, 9, 0.323955},  {5, 10, 0.377122},  {6, 0, 0.728114},
            {6, 1, 0.572922},  {6, 2, 0.600398},   {6, 3, 0.731823},
            {6, 4, 0.607078},  {6, 5, 0.548686},   {6, 6, 0.372388},
            {6, 7, 0.341927},  {6, 8, 0.342702},   {6, 9, 0.403859},
            {6, 10, 0.468459}, {7, 0, 0.475139},   {7, 1, 0.670751},
            {7, 2, 0.501308},  {7, 3, 0.599921},   {7, 4, 0.394143},
            {7, 5, 0.429276},  {7, 6, 0.561681},   {7, 7, 0.445538},
            {7, 8, 0.626099},  {7, 9, 0.786562},   {7, 10, 0.255453},
            {8, 0, 0.223515},  {8, 1, 0.331156},   {8, 2, 0.176681},
            {8, 3, 0.695574},  {8, 4, 0.178912},   {8, 5, 0.310622},
            {8, 6, 0.445796},  {8, 7, 0.446313},   {8, 8, 0.0646119},
            {8, 9, 0.545403},  {8, 10, 0.462326},  {9, 0, 0.513944},
            {9, 1, 0.743972},  {9, 2, 0.505344},   {9, 3, 0.394114},
            {9, 4, 0.234896},  {9, 5, 0.593275},   {9, 6, 0.420085},
            {9, 7, 0.405696},  {9, 8, 0.626986},   {9, 10, 0.532113},
            {10, 0, 0.352077}, {10, 1, 0.478829},  {10, 2, 0.41936},
            {10, 3, 0.759827}, {10, 4, 0.662362},  {10, 5, 0.409371},
            {10, 6, 0.126773}, {10, 7, 0.326123},  {10, 8, 0.409758},
            {10, 9, 0.383666}, {10, 10, 0.583833},
            },
      .expected = 13.7274,
  };

  test_case_t t76_1 = {
      .a = {-1, -1 - 2},
      .b = {2, 4 - 2},
      .c =
          {
            {0, 0 - 2, 0.429276},
            {0, 1 - 2, 0.561681},
            {0, 2 - 2, 0.445538},
            {0, 3 - 2, 0.626099},
            {0, 4 - 2, 0.786562},
            {1, 1 - 2, 0.445796},
            {1, 2 - 2, 0.446313},
            /* {2, 0 - 2, 0.593275}, */
              /* {2, 1 - 2, 0.420085}, */
              /* {2, 2 - 2, 0.405696}, */
              {2, 3 - 2, 0.626986},
            },
      .expected = 0.,
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
      /* &t9, */
      /* &tA, */
      /* &tB, */
      &t76,
      /* &t76_1, */
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

void DrawSystem(
    const std::set<Point>& points,
    const std::vector<Circle>& circles,
    const std::vector<LineSegment>& lines,
    const std::vector<Point>& path,
    const std::map<size_t, std::set<Point>>& dct_circle_intersections)
{
  const char* wn = "TipToe through the circle";
  const double scale = 100.;
  const int width = 900;
  const int height = 700;
  const int center_offset_x = width / 2;
  const int center_offset_y = height / 2;
  cv::Mat mat_img(height, width, CV_8UC4, cv::Scalar(255, 255, 255));

  cv::Scalar color_circle(0xe0, 0xe0, 0xe0);
  cv::Scalar color_point(0, 0, 0);
  cv::Scalar color_line(0x00, 0xff, 0x00);
  cv::Scalar color_path(0x00, 0x00, 0xff);
  cv::Scalar color_start(0xff, 0x00, 0x00);
  cv::Scalar color_end(0xff, 0xff, 0x00);
  cv::Scalar color_cc_intersection(0x00, 0xff, 0xff);

  cv::namedWindow(wn);

  for (auto& c : circles) {
    cv::circle(mat_img,
               cv::Point(c.ctr.x * scale + center_offset_x,
                         c.ctr.y * scale + center_offset_y),
               c.r * scale,
               color_circle);
  }

  cv::line(mat_img,
           cv::Point(width / 2, 0),
           cv::Point(width / 2, height),
           color_point);
  cv::line(mat_img,
           cv::Point(0, height / 2),
           cv::Point(width, height / 2),
           color_point);

  // X
  for (int x = 0; x < width / 2; x += static_cast<int>(scale)) {
    const int height = 15;
    cv::line(mat_img,
             cv::Point(x + center_offset_x, height + center_offset_y),
             cv::Point(x + center_offset_x, -height + center_offset_y),
             color_point);
    cv::line(mat_img,
             cv::Point(-x + center_offset_x, height + center_offset_y),
             cv::Point(-x + center_offset_x, -height + center_offset_y),
             color_point);
  }

  // Y
  for (int y = 0; y < height / 2; y += static_cast<int>(scale)) {
    const int width = 15;
    cv::line(mat_img,
             cv::Point(-width + center_offset_x, y + center_offset_y),
             cv::Point(width + center_offset_x, y + center_offset_y),
             color_point);
    cv::line(mat_img,
             cv::Point(-width + center_offset_x, -y + center_offset_y),
             cv::Point(width + center_offset_x, -y + center_offset_y),
             color_point);
  }

  for (auto& ls : lines) {
    cv::line(mat_img,
             cv::Point(ls.a.x * scale + center_offset_x,
                       ls.a.y * scale + center_offset_y),
             cv::Point(ls.b.x * scale + center_offset_x,
                       ls.b.y * scale + center_offset_y),
             color_line);
  }

  for (auto& p : points) {
    cv::circle(
        mat_img,
        cv::Point(p.x * scale + center_offset_x, p.y * scale + center_offset_y),
        1,
        color_point,
        -1);
  }

  for (auto& ci : dct_circle_intersections) {
    for (auto& p : ci.second) {
      cv::circle(mat_img,
                 cv::Point(p.x * scale + center_offset_x,
                           p.y * scale + center_offset_y),
                 1,
                 color_cc_intersection,
                 -1);
    }
  }

  for (size_t i = 0; i < path.size() - 1; ++i) {
    cv::Point a(path[i].x * scale + center_offset_x,
                path[i].y * scale + center_offset_y);
    cv::Point b(path[i + 1].x * scale + center_offset_x,
                path[i + 1].y * scale + center_offset_y);
    cv::line(mat_img, a, b, color_path);
  }

  cv::circle(mat_img,
             cv::Point(path.front().x * scale + center_offset_x,
                       path.front().y * scale + center_offset_y),
             1,
             color_start,
             -1);
  cv::circle(mat_img,
             cv::Point(path.back().x * scale + center_offset_x,
                       path.back().y * scale + center_offset_y),
             1,
             color_end,
             -1);

  cv::flip(mat_img, mat_img, 0);
  cv::imshow(wn, mat_img);
  while (cv::waitKey(1) != 0x1b)  // ESC
    ;
}
//////////////////////////////////////////////////////////////

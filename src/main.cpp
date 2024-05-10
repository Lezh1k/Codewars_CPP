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
  friend bool operator<(const Point& p1, const Point& p2);
};

bool operator<(const Point& p1, const Point& p2)
{
  if (std::fabs(p1.x - p2.x) <= EPS) {
    return std::round(p1.y / EPS) * EPS < std::round(p2.y / EPS) * EPS;
  }
  return std::round(p1.x / EPS) * EPS < std::round(p2.x / EPS) * EPS;
}
//////////////////////////////////////////////////////////////

struct Circle {
  Point center;
  double radius;

  Circle() : center(), radius(1.0) {}
  Circle(Point center, double radius) : center(center), radius(radius) {}
  Circle(double center_x, double center_y, double radius)
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

static void DrawSystem(const std::set<Point>& points,
                       const std::vector<Circle>& circles,
                       const std::vector<LineSegment>& lines);

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
  d = sqrt(abs(d));
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
      Point pt(c2.center.x - c1.center.x, c2.center.y - c1.center.y);
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

std::vector<Point> intersection_lc(const Line& l, const Circle& circle)
{
  double a = l.a;
  double b = l.b;
  double c = l.c;
  double r = circle.radius;

  double x0 = -a * c / (a * a + b * b);
  double y0 = -b * c / (a * a + b * b);

  if (c * c > r * r * (a * a + b * b) + EPS) {
    return std::vector<Point>();  // empty
  }

  if (abs(c * c - r * r * (a * a + b * b)) < EPS) {
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
  std::vector<Point> ips = intersection_lc(l, c1);
  for (Point& p : ips) {
    p.x += c1.center.x;
    p.y += c1.center.y;
  }
  return ips;
}
//////////////////////////////////////////////////////////////

Point intersection_ntc(const Line& l, const Circle& circle)
{
  if (abs(l.a * l.a + l.b * l.b - 1.) >= EPS) {
    throw std::invalid_argument("line is not normalized");
  }

  std::vector<Point> res;
  double cx = circle.center.x;
  double cy = circle.center.y;
  double r = circle.radius;

  double a = l.a;
  double b = l.b;
  double c = l.c - (a * -cx + b * -cy);  // shift line

  double x0 = -a * c;
  double y0 = -b * c;

  double dcr = c * c - r * r;
  assert(abs(dcr) <= EPS);  // they should be equal.

  return Point(x0 + cx, y0 + cy);
}
//////////////////////////////////////////////////////////////

bool line_segment_cross_circle(const LineSegment& ls, const Circle& c)
{
  double x1 = ls.a.x;
  double y1 = ls.a.y;
  double x2 = ls.b.x;
  double y2 = ls.b.y;
  double x3 = c.center.x;
  double y3 = c.center.y;

  // check that neither point 1 nor point 2 are inside the circle
  double d1 = sqrt((x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1));
  double d2 = sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
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

  double d = sqrt((x3 - x) * (x3 - x) + (y3 - y) * (y3 - y));
  return d < c.radius;
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
  for (Point p : dct_circle_intersections.at(c1_idx)) {
    circles_points.push_back(p);
  }

  std::sort(circles_points.begin(),
            circles_points.end(),
            [&c1](const Point& p1, const Point& p2) {
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

    /* std::cout << "circle_idx: " << c1_idx << " si: " << begin << " se: " <<
     * end */
    /*           << "\n"; */

    for (size_t ai = begin + 1; ai % circles_points.size() != end; ++ai) {
      Point a = circles_points[ai % circles_points.size()];
      for (size_t bi = ai + 1; bi % circles_points.size() != end; ++bi) {
        Point b = circles_points[bi % circles_points.size()];
        double cos_th = (a.x * b.x + a.y * b.y) / (c1.radius * c1.radius);
        double th = acos(cos_th);
        double d_ab = th * M_PI * c1.radius;
        out_graph[a].insert(Vertex(b, d_ab));
        out_graph[b].insert(Vertex(a, d_ab));
        /* std::cout << "\tai: " << ai % circles_points.size() */
        /*           << " bi: " << bi % circles_points.size() << "\n"; */
      }
    }

    i = se;
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
      double distance = sqrt(dx * dx + dy * dy);
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
  std::vector<Circle> circles = {Circle(a, 0.)};
  circles.insert(circles.end(), in_circles.begin(), in_circles.end());
  circles.push_back(Circle(b, 0.));

  std::vector<LineSegment> line_segments;
  std::set<Point> points;

  // key - index of circle
  // value - set of points on this circle
  std::map<size_t, std::set<Point>> dct_circle_points;
  // points where circle intersects with other circles
  std::map<size_t, std::set<Point>> dct_circle_intersections =
      find_all_circle_circle_intersections(circles);

#if 1
  // for debug and display only:
  for (auto& cc_intersection : dct_circle_intersections) {
    for (auto p : cc_intersection.second) {
      points.insert(p);
    }
  }
  ////////////////////
#endif

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

  for (auto it : graph) {
    std::cout << it.first.x << " " << it.first.y << "\n";
    for (auto v : it.second) {
      std::cout << "\t" << v.point.x << " " << v.point.y << " " << v.distance
                << "\n";
    }
  }

  DrawSystem(points, circles, line_segments);
  return 0.;
}
//////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
#ifdef _UNIT_TESTS_
  return main_tests(argc, argv);
#else
  (void)argc;
  (void)argv;

  Point a, b;
  std::vector<Circle> c;
  a = {-3, 1};
  b = {5.0, 0};
  c = {
      {0.0, 0.0, 2.5},
      {0.0, 2.5, 0.5},
      {2.5, 0.0, 0.5},
      /* {3.5,  1.0, 1.0}, */
      /* {3.5,  2.0, 1.3}, */
      /* {3.5, -1.7, 1.2}, */
  };

  try {
    shortest_path_length(a, b, c);
  } catch (const std::exception& exc) {
    std::cout << exc.what() << std::endl;
    return 1;
  }

  return 0;
#endif
}
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

void DrawSystem(const std::set<Point>& points,
                const std::vector<Circle>& circles,
                const std::vector<LineSegment>& lines)
{
  const char* wn = "TipToe through the circle";
  const double scale = 60.;
  const int width = 1000;
  const int height = 700;
  const int center_offset_x = width / 2;
  const int center_offset_y = height / 2;
  cv::Mat mat_img(height, width, CV_8UC4, cv::Scalar(255, 255, 255));

  cv::Scalar circle_color(0xe0, 0xe0, 0xe0);
  cv::Scalar point_color(0, 0, 0);
  cv::Scalar line_color(0x00, 0xff, 0x00);

  cv::namedWindow(wn);

  for (auto& c : circles) {
    cv::circle(mat_img,
               cv::Point(c.center.x * scale + center_offset_x,
                         c.center.y * scale + center_offset_y),
               c.radius * scale,
               circle_color,
               -1);
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
        2);
  }

  cv::flip(mat_img, mat_img, 0);
  cv::imshow(wn, mat_img);
  while (cv::waitKey(1) != 0x1b)  // ESC
    ;
}
//////////////////////////////////////////////////////////////

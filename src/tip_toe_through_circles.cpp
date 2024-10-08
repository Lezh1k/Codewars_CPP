#include "tip_toe_through_circles.h"

#include <assert.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <set>
#include <utility>
#include <vector>

static const double EPS = 1e-9;
static const double INF = 9999999999.0;
#define DEBUG 1

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
  // circle indices
  size_t a_idx;
  size_t b_idx;
  LineSegment() : a(), b(), a_idx(-1), b_idx(-1) {}
  LineSegment(Point a, Point b) : a(a), b(b), a_idx(-1), b_idx(-1) {}
  LineSegment(Point a, size_t a_idx, Point b, size_t b_idx)
      : a(a), b(b), a_idx(a_idx), b_idx(b_idx)
  {
  }
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
  // index of circle which contains this point
  size_t circle_idx;
  double priority;

  Vertex() : point(0, 0), circle_idx(-1), priority(INF) {}
  Vertex(Point p, size_t circle_idx)
      : point(p), circle_idx(circle_idx), priority(INF)
  {
  }
  Vertex(Point p, size_t circle_idx, double cost)
      : point(p), circle_idx(circle_idx), priority(cost)
  {
  }
};

inline bool operator==(const Vertex& lhs, const Vertex& rhs)
{
  return lhs.point == rhs.point;
}

inline bool operator!=(const Vertex& lhs, const Vertex& rhs)
{
  return !(lhs == rhs);
}

bool operator<(const Vertex& v1, const Vertex& v2)
{
  return v1.point < v2.point;
}

struct MinVertexPriority {
  bool operator()(const Vertex& l, const Vertex& r) const
  {
    return l.priority > r.priority;
    /* return std::round(l.priority / EPS) * EPS > */
    /*        std::round(r.priority / EPS) * EPS; */
  }
};
//////////////////////////////////////////////////////////////

struct Edge {
  Vertex src_vertex;
  Vertex dst_vertex;
  double distance;

  Edge() : src_vertex(), dst_vertex(), distance(0.) {}
  Edge(const Vertex& src_vertex, const Vertex& dst_vertex, double distance)
      : src_vertex(src_vertex), dst_vertex(dst_vertex), distance(distance)
  {
  }

  Edge(Point src_point,
       size_t src_circle_idx,
       Point dst_point,
       size_t dst_circle_idx,
       double distance)
      : src_vertex(src_point, src_circle_idx),
        dst_vertex(dst_point, dst_circle_idx),
        distance(distance)
  {
  }
};

std::ostream& operator<<(std::ostream& os, const Edge& e)
{
  os << e.src_vertex.point << " ==> " << e.dst_vertex.point << " : "
     << e.distance << ":::" << e.src_vertex.circle_idx << " "
     << e.dst_vertex.circle_idx;
  return os;
}
//////////////////////////////////////////////////////////////

typedef std::vector<Edge> graph_t;
//////////////////////////////////////////////////////////////

static void DrawSystem(
    const std::vector<Circle>& circles,
    const graph_t& graph,
    const std::vector<Point>& path,
    const std::vector<std::set<Point>>& dct_circle_intersections);

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

static bool line_segment_cross_circle(const LineSegment& ls, const Circle& c);
static std::vector<Edge> get_neighbours(
    const Vertex& vertex,
    const std::vector<Circle>& circles,
    std::vector<std::vector<LineSegment>>& dct_circle_circle_connections,
    const std::vector<std::set<Point>>& dct_circle_intersections);

static std::vector<std::set<Point>> find_all_circle_circle_intersections(
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

static double norm_atan2(double y, double x)
{
  double angle = atan2(y, x);
  if (angle < 0.)
    angle += 2 * M_PI;
  return angle;
}
//////////////////////////////////////////////////////////////

static std::vector<LineSegment> get_all_circle_circle_connections(
    size_t c1_idx,
    const std::vector<Circle>& circles)
{
  std::vector<LineSegment> result;
  const Circle& c1 = circles[c1_idx];
  for (size_t c2_idx = 0; c2_idx < circles.size(); ++c2_idx) {
    if (c2_idx == c1_idx)
      continue;

    const Circle& c2 = circles[c2_idx];
    std::vector<Line> tls = tangences(c1, c2);

    for (auto l : tls) {
      Point pc1 = intersection_ntc(l, c1);
      Point pc2 = intersection_ntc(l, c2);
      LineSegment ls(pc1, c1_idx, pc2, c2_idx);
      bool exclude = false;
      for (size_t k = 0; k < circles.size() && !exclude; ++k) {
        // we don't need to check src and dst circles
        if (k == c1_idx || k == c2_idx)
          continue;
        exclude = line_segment_cross_circle(ls, circles[k]);
      }

      if (!exclude) {
        result.push_back(ls);
      }
    }  // for l : tls
  }  // for (size_t c2_idx = 0; c2_idx < circles.size(); ++c2_idx)
  return result;
}

/**
 * This method returns all LineSegments between all other circles
 * First - point on vertex::circle, second - point on another circle
 */
static std::vector<LineSegment> get_all_circle_circle_connections(
    const Vertex& vertex,
    const std::vector<Circle>& circles)
{
  std::vector<LineSegment> res =
      get_all_circle_circle_connections(vertex.circle_idx, circles);
  for (const LineSegment& ls : res) {
    if (ls.a != ls.b)
      continue;
    if (ls.a_idx == ls.b_idx)
      continue;

    // need to process same point on different circles
    std::vector<LineSegment> extra =
        get_all_circle_circle_connections(ls.b_idx, circles);
    std::copy(extra.begin(), extra.end(), std::back_inserter(res));
  }
  return res;
}
//////////////////////////////////////////////////////////////

static std::vector<Edge> get_all_vertex_line_connections(
    const Vertex& vertex,
    const std::vector<std::vector<LineSegment>>& dct_circle_circle_connections)
{
  std::vector<Edge> result;
  size_t cidx = vertex.circle_idx;
  for (const auto& ls : dct_circle_circle_connections[cidx]) {
    if (ls.a != vertex.point) {
      continue;
    }

    double dx = ls.b.x - ls.a.x;
    double dy = ls.b.y - ls.a.y;
    double distance = std::sqrt(dx * dx + dy * dy);
    result.push_back(Edge(ls.a, ls.a_idx, ls.b, ls.b_idx, distance));
  }

  return result;
}
//////////////////////////////////////////////////////////////

static std::vector<Edge> get_all_vertex_arc_connections(
    const Vertex& vertex,
    const Circle& circle,
    const std::vector<std::vector<LineSegment>>& dct_circle_circle_connections,
    const std::vector<std::set<Point>>& dct_circle_intersections)
{
  struct point_info_t {
    Point p;
    double th;
    bool intersection;
  };

  size_t c1_idx = vertex.circle_idx;
  std::vector<point_info_t> points;
  points.reserve(dct_circle_circle_connections[c1_idx].size() +
                 dct_circle_intersections[c1_idx].size());

  // fill circles_points array with points and their angle
  for (const LineSegment& ls : dct_circle_circle_connections[c1_idx]) {
    point_info_t pi = {
        .p = ls.a,
        .th = norm_atan2(ls.a.y - circle.ctr.y, ls.a.x - circle.ctr.x),
        .intersection = false,
    };
    points.push_back(pi);
  }

  for (const Point& p : dct_circle_intersections[c1_idx]) {
    point_info_t pi = {
        .p = p,
        .th = norm_atan2(p.y - circle.ctr.y, p.x - circle.ctr.x),
        .intersection = true,
    };
    points.push_back(pi);
  }

  std::sort(points.begin(),
            points.end(),
            [](const point_info_t& p1, const point_info_t& p2) {
              return p1.th < p2.th;
            });

  // s - index of vertex.point
  size_t s = points.size();
  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].p == vertex.point) {
      s = i;
      break;
    }
  }
  if (s == points.size()) {
    // not found
    return std::vector<Edge>();
  }

  // go right until intersection or s point
  size_t r_idx = points.size();
  for (size_t i = 1; i < points.size(); ++i) {
    size_t si = (s + i) % points.size();
    if (points[si].intersection) {
      r_idx = si;
      break;
    }
  }
  // check r_idx

  // go left until intersection or s point
  size_t l_idx = points.size();
  for (size_t i = 1; i < points.size(); ++i) {
    size_t si = (s + points.size() - i) % points.size();
    if (points[si].intersection) {
      l_idx = si;
      break;
    }
  }

  std::vector<Edge> result;
  if (l_idx == points.size() || r_idx == points.size()) {
    // do calculations without intersection
    const point_info_t& a = points[s];
    double ax = a.p.x - circle.ctr.x;
    double ay = a.p.y - circle.ctr.y;

    for (size_t j = 0; j < points.size(); ++j) {
      const point_info_t& b = points[j];
      if (a.p == b.p)
        continue;

      double bx = b.p.x - circle.ctr.x;
      double by = b.p.y - circle.ctr.y;

      double cos_th = ax * bx + ay * by;
      cos_th /= circle.r * circle.r;

      double th = std::acos(cos_th);
      double d_ab = th * circle.r;
      result.push_back(
          Edge(a.p, vertex.circle_idx, b.p, vertex.circle_idx, d_ab));
    }  // for j
    return result;
  }  // calculations without intersections

  // else (calculations with intersections)
  // go left
  const point_info_t& a = points[s];
  for (size_t i = 1; i < points.size(); ++i) {
    size_t next_idx = (s + i) % points.size();
    const point_info_t& b = points[next_idx];
    if (next_idx == r_idx) {
      break;
    }

    if (a.p == b.p) {
      continue;
    }

    double d_th = b.th - a.th;
    if (d_th < 0.) {
      d_th += 2 * M_PI;
    }
    double d_ab = d_th * circle.r;

    result.push_back(
        Edge(a.p, vertex.circle_idx, b.p, vertex.circle_idx, d_ab));
  }

  // go right
  for (size_t i = 1; i < points.size(); ++i) {
    size_t next_idx = (s + points.size() - i) % points.size();
    const point_info_t& b = points[next_idx];
    if (next_idx == l_idx) {
      break;
    }

    if (a.p == b.p) {
      continue;
    }

    double d_th = a.th - b.th;
    if (d_th < 0.) {
      d_th += 2 * M_PI;
    }
    double d_ab = d_th * circle.r;
    result.push_back(
        Edge(a.p, vertex.circle_idx, b.p, vertex.circle_idx, d_ab));
  }

  return result;
}
//////////////////////////////////////////////////////////////

std::vector<Edge> get_neighbours(
    const Vertex& vertex,
    const std::vector<Circle>& circles,
    std::vector<std::vector<LineSegment>>& dct_circle_circle_connections,
    const std::vector<std::set<Point>>& dct_circle_intersections)
{
  std::vector<Edge> result;
  if (dct_circle_circle_connections[vertex.circle_idx].empty()) {
    dct_circle_circle_connections[vertex.circle_idx] =
        get_all_circle_circle_connections(vertex, circles);
  }

  std::vector<Edge> v_line_connections =
      get_all_vertex_line_connections(vertex, dct_circle_circle_connections);

  std::vector<Edge> v_arc_connections =
      get_all_vertex_arc_connections(vertex,
                                     circles[vertex.circle_idx],
                                     dct_circle_circle_connections,
                                     dct_circle_intersections);

  // join connections and return
  std::vector<Edge>& connections = v_line_connections;
  std::copy(v_arc_connections.begin(),
            v_arc_connections.end(),
            std::back_inserter(connections));

  return connections;
}
//////////////////////////////////////////////////////////////

std::vector<std::set<Point>> find_all_circle_circle_intersections(
    const std::vector<Circle>& circles)
{
  std::vector<std::set<Point>> dct_circle_intersections(circles.size(),
                                                        std::set<Point>());
  for (size_t i = 0; i < circles.size(); ++i) {
    const Circle& c1 = circles[i];
    for (size_t j = 0; j < circles.size(); ++j) {
      if (i == j)
        continue;

      const Circle& c2 = circles[j];
      std::vector<Point> cips = intersection_cc(c1, c2);
      for (const Point& cip : cips) {
        dct_circle_intersections[i].insert(cip);
        dct_circle_intersections[j].insert(cip);
      }
    }
  }

  return dct_circle_intersections;
}
//////////////////////////////////////////////////////////////

double shortest_path_length(const Point& a,
                            const Point& b,
                            const std::vector<Circle>& in_circles,
                            bool draw_system)
{
  std::vector<Circle> circles = {Circle(a.x, a.y, 0.)};
  std::copy(in_circles.begin(), in_circles.end(), std::back_inserter(circles));
  circles.push_back(Circle(b.x, b.y, 0.));

  // key - index of circle
  // value - set of points on this circle
  std::vector<std::vector<LineSegment>> dct_circle_circle_connections(
      circles.size(),
      std::vector<LineSegment>());

  // points where circle intersects with other circles
  std::vector<std::set<Point>> dct_circle_intersections =
      find_all_circle_circle_intersections(circles);

  // now we are going to implement A*
  Vertex v_start(a, 0, 0.);
  Vertex v_goal(b, circles.size() - 1);

  auto heuristics = [&v_goal](const Vertex& v) -> double {
    double dx = v_goal.point.x - v.point.x;
    double dy = v_goal.point.y - v.point.y;
    return std::sqrt(dx * dx + dy * dy);
  };

  std::priority_queue<Vertex, std::vector<Vertex>, MinVertexPriority> frontier;
  frontier.push(v_start);  // Q

  std::map<Vertex, double> cost_so_far;  // G
  cost_so_far.insert(std::make_pair(v_start, 0.));

  std::map<Vertex, Vertex> came_from;  // PARENT
  came_from.insert(std::make_pair(v_start, v_start));

  graph_t g;
  bool found = false;

  while (!frontier.empty()) {
    Vertex current = frontier.top();
    frontier.pop();

    if (current == v_goal) {
      found = true;
      break;
    }

    std::vector<Edge> neighbours = get_neighbours(current,
                                                  circles,
                                                  dct_circle_circle_connections,
                                                  dct_circle_intersections);

    for (const Edge& next : neighbours) {
      g.push_back(next);
      double new_cost = cost_so_far[current] + next.distance;
      if (cost_so_far.find(next.dst_vertex) != cost_so_far.end() &&
          new_cost >= cost_so_far[next.dst_vertex]) {
        continue;
      }

      cost_so_far[next.dst_vertex] = new_cost;
      double priority = new_cost + heuristics(next.dst_vertex);
      frontier.push(
          Vertex(next.dst_vertex.point, next.dst_vertex.circle_idx, priority));
      came_from[next.dst_vertex] = current;
    }

    /* DrawSystem(circles, g, std::vector<Point>(), dct_circle_intersections);
     */
  }

  std::vector<Point> path;
  if (found) {
    Vertex prev = v_goal;
    while (came_from[prev] != prev) {
      path.push_back(prev.point);
      prev = came_from[prev];
    }
    path.push_back(prev.point);
  }

  if (draw_system)
    DrawSystem(circles, g, path, dct_circle_intersections);
  return found ? cost_so_far[v_goal] : -1.;
}
//////////////////////////////////////////////////////////////

void DrawSystem(const std::vector<Circle>& circles,
                const graph_t& graph,
                const std::vector<Point>& path,
                const std::vector<std::set<Point>>& dct_circle_intersections)
{
  // window
  const char* wn = "TipToe through the circle";
  const double scale = 60.;
  const int width = 1000;
  const int height = 800;
  const int center_offset_x = width / 2;
  const int center_offset_y = height / 2;
  cv::Mat mat_img(height, width, CV_8UC4, cv::Scalar(255, 255, 255));

  // colors
  cv::Scalar color_circle(0xe0, 0xe0, 0xe0);
  cv::Scalar color_point(0, 0, 0);
  cv::Scalar color_line(0x00, 0xff, 0x00);
  cv::Scalar color_path(0x00, 0x00, 0xff);
  cv::Scalar color_start(0xff, 0x00, 0x00);
  cv::Scalar color_end(0xff, 0xff, 0x00);
  cv::Scalar color_cc_intersection(0x00, 0xff, 0xff);

  cv::namedWindow(wn);

  // draw all circles
  for (auto& c : circles) {
    cv::circle(mat_img,
               cv::Point(c.ctr.x * scale + center_offset_x,
                         c.ctr.y * scale + center_offset_y),
               c.r * scale,
               color_circle);
  }

  // X - axis
  cv::line(mat_img,
           cv::Point(0, height / 2),
           cv::Point(width, height / 2),
           color_point);
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

  // Y - axis
  cv::line(mat_img,
           cv::Point(width / 2, 0),
           cv::Point(width / 2, height),
           color_point);
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

  // draw all connection
  for (const Edge& e : graph) {
    cv::circle(mat_img,
               cv::Point(e.src_vertex.point.x * scale + center_offset_x,
                         e.src_vertex.point.y * scale + center_offset_y),
               1,
               color_point,
               -1);

    cv::line(mat_img,
             cv::Point(e.src_vertex.point.x * scale + center_offset_x,
                       e.src_vertex.point.y * scale + center_offset_y),
             cv::Point(e.dst_vertex.point.x * scale + center_offset_x,
                       e.dst_vertex.point.y * scale + center_offset_y),
             color_line);

    cv::circle(mat_img,
               cv::Point(e.dst_vertex.point.x * scale + center_offset_x,
                         e.dst_vertex.point.y * scale + center_offset_y),
               1,
               color_point,
               -1);
  }

  // draw all intersections
  for (auto& ci : dct_circle_intersections) {
    for (auto& p : ci) {
      cv::circle(mat_img,
                 cv::Point(p.x * scale + center_offset_x,
                           p.y * scale + center_offset_y),
                 1,
                 color_cc_intersection,
                 -1);
    }
  }

  // draw start/end points
  cv::circle(mat_img,
             cv::Point(circles.front().ctr.x * scale + center_offset_x,
                       circles.front().ctr.y * scale + center_offset_y),
             1,
             color_start,
             -1);
  cv::circle(mat_img,
             cv::Point(circles.back().ctr.x * scale + center_offset_x,
                       circles.back().ctr.y * scale + center_offset_y),
             1,
             color_end,
             -1);

  // draw path
  if (!path.empty()) {
    for (size_t i = 0; i < path.size() - 1; ++i) {
      cv::Point a(path[i].x * scale + center_offset_x,
                  path[i].y * scale + center_offset_y);
      cv::Point b(path[i + 1].x * scale + center_offset_x,
                  path[i + 1].y * scale + center_offset_y);
      cv::line(mat_img, a, b, color_path);
    }
  }

  cv::flip(mat_img, mat_img, 0);
  cv::imshow(wn, mat_img);
  while (cv::waitKey(1) != 0x1b)  // ESC
    ;
}
//////////////////////////////////////////////////////////////

int run_all_test_cases(bool draw_system)
{
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
      .a = {1 - 5, 1 - 5},
      .b = {9 - 5, 9 - 5},
      .c =
          {
            {0 - 5, 0 - 5, 0.64115},    {0 - 5, 1 - 5, 0.132413},
            {0 - 5, 2 - 5, 0.360349},   {0 - 5, 3 - 5, 0.324987},
            {0 - 5, 4 - 5, 0.291204},   {0 - 5, 5 - 5, 0.482743},
            {0 - 5, 6 - 5, 0.357549},   {0 - 5, 7 - 5, 0.472708},
            {0 - 5, 8 - 5, 0.487758},   {0 - 5, 9 - 5, 0.502299},
            {0 - 5, 10 - 5, 0.291764},  {1 - 5, 0 - 5, 0.253298},
            {1 - 5, 2 - 5, 0.289496},   {1 - 5, 3 - 5, 0.487209},
            {1 - 5, 4 - 5, 0.205027},   {1 - 5, 5 - 5, 0.710598},
            {1 - 5, 6 - 5, 0.355356},   {1 - 5, 7 - 5, 0.474757},
            {1 - 5, 8 - 5, 0.15398},    {1 - 5, 9 - 5, 0.552585},
            {1 - 5, 10 - 5, 0.354647},  {2 - 5, 0 - 5, 0.374451},
            {2 - 5, 1 - 5, 0.417562},   {2 - 5, 2 - 5, 0.80148},
            {2 - 5, 3 - 5, 0.226192},   {2 - 5, 4 - 5, 0.256702},
            {2 - 5, 5 - 5, 0.355266},   {2 - 5, 6 - 5, 0.409288},
            {2 - 5, 7 - 5, 0.327123},   {2 - 5, 8 - 5, 0.302255},
            {2 - 5, 9 - 5, 0.331616},   {2 - 5, 10 - 5, 0.116894},
            {3 - 5, 0 - 5, 0.461943},   {3 - 5, 1 - 5, 0.665481},
            {3 - 5, 2 - 5, 0.472481},   {3 - 5, 3 - 5, 0.184833},
            {3 - 5, 4 - 5, 0.332489},   {3 - 5, 5 - 5, 0.454663},
            {3 - 5, 6 - 5, 0.368843},   {3 - 5, 7 - 5, 0.273874},
            {3 - 5, 8 - 5, 0.499421},   {3 - 5, 9 - 5, 0.282398},
            {3 - 5, 10 - 5, 0.393656},  {4 - 5, 0 - 5, 0.303524},
            {4 - 5, 1 - 5, 0.395856},   {4 - 5, 2 - 5, 0.689242},
            {4 - 5, 3 - 5, 0.347418},   {4 - 5, 4 - 5, 0.366592},
            {4 - 5, 5 - 5, 0.283776},   {4 - 5, 6 - 5, 0.293723},
            {4 - 5, 7 - 5, 0.598069},   {4 - 5, 8 - 5, 0.327934},
            {4 - 5, 9 - 5, 0.693837},   {4 - 5, 10 - 5, 0.371845},
            {5 - 5, 0 - 5, 0.530865},   {5 - 5, 1 - 5, 0.485497},
            {5 - 5, 2 - 5, 0.539198},   {5 - 5, 3 - 5, 0.21921},
            {5 - 5, 4 - 5, 0.373822},   {5 - 5, 5 - 5, 0.621798},
            {5 - 5, 6 - 5, 0.344001},   {5 - 5, 7 - 5, 0.498881},
            {5 - 5, 8 - 5, 0.314385},   {5 - 5, 9 - 5, 0.323955},
            {5 - 5, 10 - 5, 0.377122},  {6 - 5, 0 - 5, 0.728114},
            {6 - 5, 1 - 5, 0.572922},   {6 - 5, 2 - 5, 0.600398},
            {6 - 5, 3 - 5, 0.731823},   {6 - 5, 4 - 5, 0.607078},
            {6 - 5, 5 - 5, 0.548686},   {6 - 5, 6 - 5, 0.372388},
            {6 - 5, 7 - 5, 0.341927},   {6 - 5, 8 - 5, 0.342702},
            {6 - 5, 9 - 5, 0.403859},   {6 - 5, 10 - 5, 0.468459},
            {7 - 5, 0 - 5, 0.475139},   {7 - 5, 1 - 5, 0.670751},
            {7 - 5, 2 - 5, 0.501308},   {7 - 5, 3 - 5, 0.599921},
            {7 - 5, 4 - 5, 0.394143},   {7 - 5, 5 - 5, 0.429276},
            {7 - 5, 6 - 5, 0.561681},   {7 - 5, 7 - 5, 0.445538},
            {7 - 5, 8 - 5, 0.626099},   {7 - 5, 9 - 5, 0.786562},
            {7 - 5, 10 - 5, 0.255453},  {8 - 5, 0 - 5, 0.223515},
            {8 - 5, 1 - 5, 0.331156},   {8 - 5, 2 - 5, 0.176681},
            {8 - 5, 3 - 5, 0.695574},   {8 - 5, 4 - 5, 0.178912},
            {8 - 5, 5 - 5, 0.310622},   {8 - 5, 6 - 5, 0.445796},
            {8 - 5, 7 - 5, 0.446313},   {8 - 5, 8 - 5, 0.0646119},
            {8 - 5, 9 - 5, 0.545403},   {8 - 5, 10 - 5, 0.462326},
            {9 - 5, 0 - 5, 0.513944},   {9 - 5, 1 - 5, 0.743972},
            {9 - 5, 2 - 5, 0.505344},   {9 - 5, 3 - 5, 0.394114},
            {9 - 5, 4 - 5, 0.234896},   {9 - 5, 5 - 5, 0.593275},
            {9 - 5, 6 - 5, 0.420085},   {9 - 5, 7 - 5, 0.405696},
            {9 - 5, 8 - 5, 0.626986},   {9 - 5, 10 - 5, 0.532113},
            {10 - 5, 0 - 5, 0.352077},  {10 - 5, 1 - 5, 0.478829},
            {10 - 5, 2 - 5, 0.41936},   {10 - 5, 3 - 5, 0.759827},
            {10 - 5, 4 - 5, 0.662362},  {10 - 5, 5 - 5, 0.409371},
            {10 - 5, 6 - 5, 0.126773},  {10 - 5, 7 - 5, 0.326123},
            {10 - 5, 8 - 5, 0.409758},  {10 - 5, 9 - 5, 0.383666},
            {10 - 5, 10 - 5, 0.583833},
            },
      .expected = 13.7274470521,
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
            {2, 0 - 2, 0.593275},
            {2, 1 - 2, 0.420085},
            {2, 2 - 2, 0.405696},
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
      double act = shortest_path_length(ptc->a, ptc->b, ptc->c, draw_system);
      std::cout << std::setprecision(12) << act << " == " << ptc->expected;
      std::cout << "\t" << (std::fabs(act - ptc->expected) <= EPS) << "\n";
    } catch (const std::exception& exc) {
      std::cout << exc.what() << std::endl;
      return 1;
    }
  }

  return 0;
}

#ifndef _GAME_MODEL_H
#define _GAME_MODEL_H

#include "vertex.h"
#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <unordered_set>

using vertex_ptr = std::shared_ptr<vertex>;

struct VertexQueueSorter
    : public std::binary_function<const vertex_ptr &, const vertex_ptr &,
                                  bool> {
  bool operator()(const vertex_ptr &arg1, const vertex_ptr &arg2) {
    return arg1->cost() + arg1->heuristics() >
           arg2->cost() + arg2->heuristics();
  }
};

struct VertexSetComparator
    : public std::binary_function<const vertex_ptr &, const vertex_ptr &,
                                  bool> {
  bool operator()(const vertex_ptr &l, const vertex_ptr &r) const {
    return l->state() < r->state();
  }
};
//////////////////////////////////////////////////////////////

class Game {
private:
  int m_field_size;
  state_t m_start_state;
  state_t m_goal_state;

  bool has_solution() const;
  bool is_goal(const vertex_ptr game_state) const;
  std::vector<vertex_ptr> get_neighbours(const vertex_ptr &game_state) const;

  void a_star_add_neighbours(
      const vertex_ptr &game_state,
      std::set<vertex_ptr, VertexSetComparator> &visited,
      std::priority_queue<vertex_ptr, std::deque<vertex_ptr>, VertexQueueSorter>
          &vrtx_queue);
  std::pair<int, vertex_ptr> ida_star_search(vertex_ptr vrtx, int bound);

public:
  Game() = delete;
  explicit Game(const std::vector<std::vector<int>> &start_state);
  ~Game() = default;

  std::vector<int> decision_states(const vertex_ptr &goal_vrtx) const;
  std::vector<int> find_solution_a_star();
  std::vector<int> find_solution_ida_star();
};

#endif

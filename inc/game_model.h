#ifndef _GAME_MODEL_H
#define _GAME_MODEL_H

#include "vertex.h"
#include <algorithm>
#include <deque>
#include <functional>
#include <queue>
#include <set>
#include <stack>
#include <unordered_set>

struct VertexQueueSorter
    : public std::binary_function<const vertex *, const vertex *, bool> {
  bool operator()(const vertex *arg1, const vertex *arg2) {
    return arg1->cost() + arg1->heuristics() >
           arg2->cost() + arg2->heuristics();
  }
};

struct VertexSetComparator
    : public std::binary_function<const vertex *, const vertex *, bool> {
  bool operator()(const vertex *l, const vertex *r) const {
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
  bool is_goal(const vertex *game_state) const;
  std::vector<vertex *> get_neighbours(const vertex *game_state) const;

  void a_star_add_neighbours(
      const vertex *game_state,
      std::set<vertex *, VertexSetComparator> &visited,
      std::priority_queue<vertex *, std::deque<vertex *>, VertexQueueSorter>
          &vrtx_queue);
  std::pair<int, vertex *> ida_star_search(vertex *vrtx, int bound);

public:
  Game() = delete;
  explicit Game(const std::vector<std::vector<int>> &start_state);
  ~Game() = default;

  std::vector<int> decision_states(const vertex *goal_vrtx) const;
  std::vector<int> find_solution_a_star();
  std::vector<int> find_solution_ida_star();
};

#endif

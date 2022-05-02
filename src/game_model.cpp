#include "game_model.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>

using defer = std::shared_ptr<void>;

Game::Game(const std::vector<std::vector<int>> &start_state)
    : m_field_size(start_state.size()), m_start_state(start_state),
      m_goal_state(start_state.size()) {
  // init goal state
  int k = 0;
  for (int r = 0; r < m_field_size; ++r) {
    m_goal_state[r].resize(m_field_size);
    for (int c = 0; c < m_field_size; ++c) {
      m_goal_state[r][c] = ++k;
    }
  }
  m_goal_state[m_field_size - 1][m_field_size - 1] = 0;
}
//////////////////////////////////////////////////////////////

bool Game::is_goal(const vertex *game_state) const {
  return m_goal_state == game_state->state();
}

std::vector<vertex *> Game::get_neighbours(const vertex *game_state) const {
  int zr = 0;
  int zc = 0; // zero row and col
  for (int r = 0; r < m_field_size; ++r) {
    for (int c = 0; c < m_field_size; ++c) {
      if ((*game_state)(r, c))
        continue;
      zr = r;
      zc = c;
      break;
    }
  }

  const int delta_r[] = {-1, 0, 1, 0};
  const int delta_c[] = {0, -1, 0, 1};

  std::vector<vertex *> res;
  res.reserve(4);
  for (int k = 0; k < 4; k++) {
    int nr = zr + delta_r[k];
    int nc = zc + delta_c[k];
    if (nr < 0 || nc < 0 || nr >= m_field_size || nc >= m_field_size)
      continue;

    vertex *nVertex = new vertex(game_state);
    (*nVertex)(nr, nc) = 0;
    (*nVertex)(zr, zc) = (*game_state)(nr, nc);
    nVertex->set_tile((*game_state)(nr, nc));
    res.push_back(nVertex);
  } // end for
  return res;
}
//////////////////////////////////////////////////////////////////////////

void Game::a_star_add_neighbours(
    const vertex *game_state, std::set<vertex *, VertexSetComparator> &visited,
    std::priority_queue<vertex *, std::deque<vertex *>, VertexQueueSorter>
        &vrtx_queue) {
  auto neighbours = get_neighbours(game_state);
  for (auto nVertex : neighbours) {
    if (visited.find(nVertex) != visited.end()) {
      delete nVertex;
      continue;
    }
    nVertex->recalculate_heuristics();
    vrtx_queue.push(nVertex);
    visited.insert(nVertex);
  } // end for
}

std::vector<int> Game::find_solution_a_star() {
  if (!has_solution()) {
    return std::vector<int>({0});
  }

  std::priority_queue<vertex *, std::deque<vertex *>, VertexQueueSorter>
      vrtx_queue;
  std::set<vertex *, VertexSetComparator> visited;
  vertex *start_vrtx = new vertex(m_start_state);
  vrtx_queue.push(start_vrtx);
  visited.insert(start_vrtx);

  defer d_visited(nullptr, [&visited](...) {
    for (auto &it : visited)
      delete it;
  });

  while (!vrtx_queue.empty()) {
    vertex *top_v = vrtx_queue.top();
    vrtx_queue.pop();
    if (!is_goal(top_v)) {
      a_star_add_neighbours(top_v, visited, vrtx_queue);
      continue;
    }
    return decision_states(top_v);
  }
  return std::vector<int>({0}); // nothing
}
//////////////////////////////////////////////////////////////

std::pair<int, vertex *> Game::ida_star_search(vertex *vrtx, int bound) {
  vrtx->recalculate_heuristics();
  int f = vrtx->heuristics() + vrtx->cost();
  if (f > bound) {
    return std::make_pair(f, nullptr);
  }

  if (is_goal(vrtx)) {
    return std::make_pair(f, vrtx);
  }

  int min = std::numeric_limits<int>::max();
  std::vector<vertex *> neighbours = get_neighbours(vrtx);
  for (auto &n : neighbours) {
    auto r = ida_star_search(n, bound);
    if (r.second != nullptr) {
      for (auto &nn : neighbours) {
        if (n != nn) delete nn;
      }
      return r;
    }
    if (r.first < min)
      min = r.first;
  }

  for (auto &n : neighbours)
    delete n;
  return std::make_pair(min, nullptr);
}
//////////////////////////////////////////////////////////////////////////

std::vector<int> Game::find_solution_ida_star() {
  if (!has_solution()) {
    return std::vector<int>({0});
  }

  const vertex *solution_node = nullptr;
  vertex *root = new vertex(m_start_state); // not shared ptr, see d_path
  defer d_path(nullptr, [&solution_node](...) {
    while (solution_node != nullptr) {
      const vertex *tmp = solution_node;
      solution_node = solution_node->parent();
      delete tmp;
    }
  });

  root->recalculate_heuristics();
  int bound = root->heuristics() /*+ root->cost(), but cost is zero here*/;
  while (solution_node == nullptr) {
    std::pair<int, vertex *> t = ida_star_search(root, bound);
    if (t.second != nullptr) {
      solution_node = t.second;
    }
    bound = t.first;
  }
  return decision_states(solution_node);
}
//////////////////////////////////////////////////////////////////////////

std::vector<int> Game::decision_states(const vertex *goal_vrtx) const {
  if (goal_vrtx == nullptr)
    return std::vector<int>({0});
  std::vector<int> res;
  while (goal_vrtx && goal_vrtx->parent() != nullptr) {
    res.push_back(goal_vrtx->tile());
    goal_vrtx = goal_vrtx->parent();
  }
  std::reverse(res.begin(), res.end());
  return res;
}
//////////////////////////////////////////////////////////////////////////

bool Game::has_solution() const {
  vertex *start_vrtx = new vertex(m_start_state);
  int N = m_field_size * m_field_size;
  std::vector<int> arr(N);

  int zr = 0;
  for (int ri = 0; ri < m_field_size; ++ri) {
    for (int ci = 0; ci < m_field_size; ++ci) {
      arr[ri * m_field_size + ci] = (*start_vrtx)(ri, ci);
      if ((*start_vrtx)(ri, ci) == 0)
        zr = ri;
    }
  }
  ++zr; // because it starts from 1
  delete start_vrtx;

  // TODO calculate with merge sort
  int inv = 0;
  for (int i = 0; i < N; ++i) {
    if (arr[i] == 0)
      continue;

    for (int j = i + 1; j < N; ++j) {
      if (arr[j] == 0)
        continue;
      inv += (arr[i] > arr[j] ? 1 : 0);
    }
  }

  if (m_field_size % 2 == 0) {
    inv += zr;
  }
  return inv % 2 == 0; // cause boards parity is even
}
//////////////////////////////////////////////////////////////////////////

#include "game_model.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>

using defer = std::shared_ptr<void>;
Game::Game(const std::vector<std::vector<int32_t>> &start_state)
    : m_field_size(start_state.size()), m_start_state(start_state),
      m_goal_state(start_state.size()) {
  // init goal state
  int32_t k = 0;
  for (int32_t r = 0; r < m_field_size; ++r) {
    m_goal_state[r].resize(m_field_size);
    for (int32_t c = 0; c < m_field_size; ++c) {
      m_goal_state[r][c] = ++k;
    }
  }
  m_goal_state[m_field_size - 1][m_field_size - 1] = 0;
}
//////////////////////////////////////////////////////////////

bool Game::is_goal(const vertex_ptr game_state) const {
  return m_goal_state == game_state->state();
}

std::vector<vertex_ptr>
Game::get_neighbours(const vertex_ptr &game_state) const {
  vertex_move_direction deltas[] = {
    VD_UP, VD_LEFT, VD_DOWN, VD_RIGHT
  };
  std::vector<vertex_ptr> res;
  res.reserve(4);
  for (int32_t k = 0; k < 4; k++) {
    if (!game_state->zero_move_is_possible(deltas[k]))
      continue;
    vertex_ptr nVertex = std::make_shared<vertex>(game_state);
    nVertex->zero_move(deltas[k]);
    res.push_back(nVertex);
  } // end for
  return res;
}
//////////////////////////////////////////////////////////////////////////

void Game::a_star_add_neighbours(
    const vertex_ptr &game_state,
    std::set<vertex_ptr, VertexSetComparator> &visited,
    std::priority_queue<vertex_ptr, std::deque<vertex_ptr>, VertexQueueSorter>
        &vrtx_queue) {
  auto neighbours = get_neighbours(game_state);
  for (auto nVertex : neighbours) {
    if (visited.find(nVertex) != visited.end()) {
      continue;
    }
    nVertex->recalculate_heuristics();
    vrtx_queue.push(nVertex);
    visited.insert(nVertex);
  } // end for
}

std::vector<int32_t> Game::find_solution_a_star() {
  if (!has_solution()) {
    return std::vector<int32_t>({0});
  }

  std::priority_queue<vertex_ptr, std::deque<vertex_ptr>, VertexQueueSorter>
      vrtx_queue;
  std::set<vertex_ptr, VertexSetComparator> visited;
  vertex_ptr start_vrtx = std::make_shared<vertex>(m_start_state);
  vrtx_queue.push(start_vrtx);
  visited.insert(start_vrtx);

  while (!vrtx_queue.empty()) {
    vertex_ptr top_v = vrtx_queue.top();
    vrtx_queue.pop();
    if (!is_goal(top_v)) {
      a_star_add_neighbours(top_v, visited, vrtx_queue);
      continue;
    }
    return decision_states(top_v);
  }
  return std::vector<int32_t>({0}); // nothing
}
//////////////////////////////////////////////////////////////

std::pair<int32_t, vertex_ptr> Game::ida_star_search(vertex_ptr vrtx, int32_t bound) {
  vrtx->recalculate_heuristics();
  int32_t f = vrtx->heuristics() + vrtx->cost();
  if (f > bound) {
    return std::make_pair(f, nullptr);
  }

  if (is_goal(vrtx)) {
    return std::make_pair(f, vrtx);
  }

  int32_t min = std::numeric_limits<int32_t>::max();
  std::vector<vertex_ptr> neighbours = get_neighbours(vrtx);
  for (auto &n : neighbours) {
    auto r = ida_star_search(n, bound);
    if (r.second != nullptr) {
      return r;
    }
    if (r.first < min) {
      min = r.first;
    }
  }
  return std::make_pair(min, nullptr);
}

std::vector<vertex_coord>
Game::bfs(const std::vector<std::vector<bool>> &lst_in_place,
          const vertex_coord &src,
          const vertex_coord &dst)
{
  std::queue<vertex_coord> q;
  std::vector<std::vector<bool>> used(m_field_size);
  std::vector<std::vector<vertex_coord>> parents(m_field_size);
  for (int32_t i = 0; i < m_field_size; ++i) {
    used[i].resize(m_field_size);
    parents[i].resize(m_field_size);
  }

  q.push(src);
  used[src.row][src.col] = true;
  parents[src.row][src.col] = vertex_coord(-1,-1);

  while (!q.empty()) {
    vertex_coord v = q.front();
    q.pop();

    std::vector<vertex_coord> neigbors;
    neigbors.reserve(4);

    const int32_t delta_r[] = {-1, 0, 1, 0};
    const int32_t delta_c[] = {0, -1, 0, 1};

    for (int k = 0; k < 4; ++k) {
      int32_t nr = v.row + delta_r[k];
      int32_t nc = v.col + delta_c[k];

      if (nr >= m_field_size || nc >= m_field_size ||
          nr < 0 || nc < 0)
        continue; //out of bounds

      if (lst_in_place[nr][nc])
        continue;

      if (!used[nr][nc]) {
        used[nr][nc] = true;
        q.push(vertex_coord(nr,nc));
        parents[nr][nc] = v;
      }
    }
  }

  std::vector<vertex_coord> path;
  if (!used[dst.row][dst.col])
    return path; // not found! we will not be here in final solution

  for (vertex_coord v = dst; v != src; v = parents[v.row][v.col]) {
    path.push_back(v);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<int32_t>
Game::move_not_empty_cell(vertex_ptr vrtx,
                          std::vector<std::vector<bool> > &lst_in_place,
                          const vertex_coord &src,
                          const vertex_coord &dst)
{
  /*  VD_UP = 0,  VD_LEFT,   VD_DOWN,   VD_RIGHT*/
  if (src == dst)
    return std::vector<int32_t>(); // return nothing, cause src is already in place!

  std::vector<int32_t> res;
  std::vector<vertex_coord> src_dst_path = bfs(lst_in_place, src, dst);
  if (src_dst_path.empty())
    return res; //todo remove after all fixes

  lst_in_place[src.row][src.col] = true;
  std::vector<vertex_coord> zero_p1_path = bfs(lst_in_place,
                                               vrtx->zero_point(),
                                               src_dst_path.front());
  lst_in_place[src.row][src.col] = false;

  for (auto &zp : zero_p1_path) {
    int dr = vrtx->zero_point().row - zp.row;
    int dc = vrtx->zero_point().col - zp.col;
    if (dr == 0 && dc == 0)
      throw std::logic_error("wrong path!");
    if (dr != 0 && dc != 0)
      throw std::logic_error("impossible move");
    vertex_move_direction dir;
    if (dr != 0) {
      dir = dr > 0 ? VD_UP : VD_DOWN;
    }
    if (dc != 0) {
      dir = dc > 0 ? VD_LEFT : VD_RIGHT;
    }
    res.push_back(vrtx->zero_move(dir));
  }
  /****/



  return res;
}
//////////////////////////////////////////////////////////////

std::vector<int32_t> Game::find_solution_ida_star() {
  if (!has_solution()) {
    return std::vector<int32_t>({0});
  }

  vertex_ptr solution_node = nullptr;
  vertex_ptr root = std::make_shared<vertex>(m_start_state);

  root->recalculate_heuristics();
  int32_t bound = root->heuristics() /*+ root->cost(), but cost is zero here*/;
  while (solution_node == nullptr) {
    std::pair<int32_t, vertex_ptr> t = ida_star_search(root, bound);
    if (t.second != nullptr) {
      solution_node = t.second;
    }
    bound = t.first;
  }
  return decision_states(solution_node);
}
//////////////////////////////////////////////////////////////

std::vector<int32_t> Game::find_solution_by_strategy()
{
  if (!has_solution()) {
    return std::vector<int32_t>({0});
  }

  std::vector<int32_t> res;
  std::shared_ptr<vertex> vrtx =
      std::make_shared<vertex>(m_start_state);
  std::vector<std::vector<bool>> lst_in_place(m_field_size);
  for (auto &lip : lst_in_place) lip.resize(m_field_size);

  auto print_val_and_coords = [=](int val,
      const vertex_coord &src_coord, const vertex_coord &dst_coord) {
    std::cout << "\t" << val <<
                 ":s[" << src_coord.row << "|" << src_coord.col << "]" <<
                 ":d[" << dst_coord.row << "|" << dst_coord.col << "]\n";
  };

  auto append_path_to_res = [this, &vrtx, &res, &lst_in_place](
      const vertex_coord &src_coord,
      const vertex_coord &dst_coord) {
    std::vector<int32_t> path =
        move_not_empty_cell(vrtx, lst_in_place, src_coord, dst_coord);
    for (auto &pp : path)
      res.push_back(pp);
    lst_in_place[dst_coord.row][dst_coord.col] = true;
  };

  for (int32_t n = 0; n < m_field_size - 2; ++n) {
    int start_val = n * (m_field_size + 1) + 1;
    int subfield_size = m_field_size - n;

    std::cout << "\nrows:\n";
    // fill dst row except last 2 elements
    for (int32_t c = 0; c < subfield_size - 2; ++c) {
      int c_val = start_val + c;
      vertex_coord cc = vrtx->find_val_coord(c_val);
      vertex_coord dc(n, n + c);
      print_val_and_coords(c_val, cc, dc);
      append_path_to_res(cc, dc);
    }

    // fill last 2 elements
    int c_pen_val = start_val + subfield_size - 2;
    int c_last_val = start_val + subfield_size - 1;
    vertex_coord cc_pen = vrtx->find_val_coord(c_pen_val);
    vertex_coord dc_pen(n, n + subfield_size - 1); //Place to LAST (top right corner)
    print_val_and_coords(c_pen_val, cc_pen, dc_pen);

    // if top right corner has last value in row - just move it 2 cells down
    if ((*vrtx)(dc_pen) == c_last_val) {
      vertex_coord tmp_dc_last(dc_pen.row+2, dc_pen.col);
      append_path_to_res(dc_pen, tmp_dc_last);
      lst_in_place[tmp_dc_last.row][tmp_dc_last.col] = false;
    }
    append_path_to_res(cc_pen, dc_pen);

    vertex_coord cc_last = vrtx->find_val_coord(c_last_val);
    vertex_coord dc_last(n+1, n + subfield_size - 1); //Place to row+1 under last!
    print_val_and_coords(c_last_val, cc_last, dc_last);
    append_path_to_res(cc_last, dc_last);

    // TODO place empty cell under dc_last and apply
    // {left, up, up, right, down} moves
    /********************************/

    std::cout << "\ncols:\n";
    // fill dst col
    for (int32_t r = 1; r < subfield_size - 2; ++r) {
      int r_val = start_val + r * (m_field_size);
      vertex_coord cr = vrtx->find_val_coord(r_val);
      vertex_coord dc(n + r, n);
      print_val_and_coords(r_val, cr, dc);
      append_path_to_res(cr, dc);
      lst_in_place[dc.row][dc.col] = true;
    }

    int r_pen_val = start_val + (subfield_size - 2) * (m_field_size);
    int r_last_val = start_val + (subfield_size - 1) * (m_field_size);
    vertex_coord cr_pen = vrtx->find_val_coord(r_pen_val);
    vertex_coord dr_pen(n + subfield_size - 1, n); //Place to LAST (left bottom corner)
    print_val_and_coords(r_pen_val, cr_pen, dr_pen);

    // if bottom left corner has last value in col - just move it 2 cells right
    if ((*vrtx)(dr_pen) == r_last_val) {
      vertex_coord tmp_dr_last(dr_pen.row, dr_pen.col+2);
      append_path_to_res(dr_pen, tmp_dr_last);
    }
    append_path_to_res(cr_pen, dr_pen);
    lst_in_place[dr_pen.row][dr_pen.col] = true;

    vertex_coord cr_last = vrtx->find_val_coord(r_last_val);
    vertex_coord dr_last(n + subfield_size - 1, n+1); //Place to col+1 right to last!
    print_val_and_coords(r_last_val, cr_last, dr_last);
    append_path_to_res(cr_last, dr_last);
    lst_in_place[dr_last.row][dr_last.col] = true;
    // TODO place empty cell right to dc_last and apply
    // {up, left, left, down, right} moves
    /********************************/
  }

  return std::vector<int32_t>({0});
}
//////////////////////////////////////////////////////////////////////////

std::vector<int32_t> Game::decision_states(const vertex_ptr &goal_vrtx) const {
  if (goal_vrtx == nullptr)
    return std::vector<int32_t>({0});
  std::vector<int32_t> res;
  const vertex *g = goal_vrtx.get();
  while (g && g->parent() != nullptr) {
    res.push_back(g->tile());
    g = g->parent().get();
  }
  std::reverse(res.begin(), res.end());
  return res;
}
//////////////////////////////////////////////////////////////////////////

bool Game::has_solution() const {
  int32_t N = m_field_size * m_field_size;
  std::vector<int32_t> arr(N);

  int32_t zr = 0;
  for (int32_t ri = 0; ri < m_field_size; ++ri) {
    for (int32_t ci = 0; ci < m_field_size; ++ci) {
      arr[ri * m_field_size + ci] = m_start_state[ri][ci];
      if (m_start_state[ri][ci] == 0)
        zr = ri;
    }
  }
  ++zr; // because it starts from 1

  // TODO calculate with merge sort
  int32_t inv = 0;
  for (int32_t i = 0; i < N; ++i) {
    if (arr[i] == 0)
      continue;

    for (int32_t j = i + 1; j < N; ++j) {
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

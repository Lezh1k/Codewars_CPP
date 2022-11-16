#include "game_model.h"
#include <algorithm>
#include <iostream>
#include <queue>

Game::Game(const std::vector<std::vector<int32_t>> &start_state)
    : m_field_size(start_state.size()),
      m_start_state(start_state),
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

bool Game::is_goal(const vertex &game_state) const {
  return m_goal_state == game_state.state();
}
//////////////////////////////////////////////////////////////

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
//////////////////////////////////////////////////////////////

std::vector<int32_t>
Game::move_not_empty_cell(vertex &vrtx,
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

  vertex_coord next = src;
  for (auto &pp : src_dst_path) {
    lst_in_place[next.row][next.col] = true;
    std::vector<vertex_coord> zero_p1_path = bfs(lst_in_place,
                                                 vrtx.zero_point(),
                                                 pp);
    lst_in_place[next.row][next.col] = false;
    zero_p1_path.push_back(next);
    auto zero_moves = move_empty_cell(vrtx, zero_p1_path);
    for (auto &zm : zero_moves)
      res.push_back(zm);
    next = pp;
  } // for pp in src_dst_path
  return res;
}

std::vector<int32_t>
Game::move_empty_cell(vertex &vrtx,
                      const std::vector<vertex_coord> &path)
{
  std::vector<int32_t> res;
  res.reserve(path.size());

  for (auto const &zp : path) {
    int dr = vrtx.zero_point().row - zp.row;
    int dc = vrtx.zero_point().col - zp.col;

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
    res.push_back(vrtx.zero_move(dir));
  } // for zp in path
  return res;
}
//////////////////////////////////////////////////////////////

std::vector<int32_t> Game::find_solution_by_strategy()
{
  if (!has_solution()) {
    return std::vector<int32_t>({0});
  }

  std::vector<int32_t> res;
  vertex vrtx(m_start_state);
  std::vector<std::vector<bool>> lst_in_place(m_field_size);
  for (auto &lip : lst_in_place)
    lip.resize(m_field_size);

  auto append_path_to_res = [this, &vrtx, &res, &lst_in_place](
      const vertex_coord &src_coord,
      const vertex_coord &dst_coord,
      bool set_in_place_flag = true) {
    std::vector<int32_t> path =
        move_not_empty_cell(vrtx, lst_in_place, src_coord, dst_coord);
    for (auto &pp : path)
      res.push_back(pp);
//    std::cout << vrtx << std::endl;
    lst_in_place[dst_coord.row][dst_coord.col] = set_in_place_flag;
  };

  for (int32_t n = 0; n < m_field_size - 2; ++n) {
    int start_val = n * (m_field_size + 1) + 1;
    int subfield_size = m_field_size - n;

    // fill dst row except last 2 elements
    for (int32_t c = 0; c < subfield_size - 2; ++c) {
      int c_val = start_val + c;
      vertex_coord cc = vrtx.find_val_coord(c_val);
      vertex_coord dc(n, n + c);
      append_path_to_res(cc, dc);
    }

    // fill last 2 elements
    int c_pen_val = start_val + subfield_size - 2;
    int c_last_val = start_val + subfield_size - 1;
    vertex_coord dc_pen(n, n + subfield_size - 1); //Place to LAST (top right corner)
    vertex_coord cc_pen = vrtx.find_val_coord(c_pen_val);
    vertex_coord cc_last = vrtx.find_val_coord(c_last_val);
    vertex_coord dc_last(n+1, n + subfield_size - 1); //Place to row+1 under last!

    // if top right corner has last value in row - just move it 2 cells down
    if (cc_last.row == dc_pen.row ||
        cc_last.col == dc_pen.col ||
        (cc_last.row == dc_pen.row+1 && cc_last.col == dc_pen.col-1)) { // vrtx(dc_pen) == c_last_val) {
      vertex_coord tmp_dc_last(dc_pen.row+2, dc_pen.col);
      append_path_to_res(cc_last, tmp_dc_last, false);
      cc_pen = vrtx.find_val_coord(c_pen_val);
    }    
    append_path_to_res(cc_pen, dc_pen);
    cc_last = vrtx.find_val_coord(c_last_val);
    append_path_to_res(cc_last, dc_last);

    vertex_coord ccz_dst(dc_pen.row, dc_pen.col-1);
    std::vector<vertex_coord> ccz_path = bfs(lst_in_place, vrtx.zero_point(), ccz_dst);
    std::vector<int> ccz_moves = move_empty_cell(vrtx, ccz_path);
    for (auto &m : ccz_moves)
      res.push_back(m);
    res.push_back(vrtx.zero_move(VD_RIGHT));
    res.push_back(vrtx.zero_move(VD_DOWN));
    lst_in_place[dc_last.row][dc_last.col] = false;
//    std::cout << vrtx << std::endl;
    /********************************/

    // fill dst col
    for (int32_t r = 1; r < subfield_size - 2; ++r) {
      int r_val = start_val + r * (m_field_size);
      vertex_coord cr = vrtx.find_val_coord(r_val);
      vertex_coord dc(n + r, n);
      append_path_to_res(cr, dc);
    }

    int r_pen_val = start_val + (subfield_size - 2) * (m_field_size);
    int r_last_val = start_val + (subfield_size - 1) * (m_field_size);
    vertex_coord dr_pen(n + subfield_size - 1, n); //Place to LAST (left bottom corner)
    vertex_coord cr_pen = vrtx.find_val_coord(r_pen_val);
    vertex_coord cr_last = vrtx.find_val_coord(r_last_val);
    vertex_coord dr_last(n + subfield_size - 1, n+1); //Place to col+1 right to last!

    // if bottom left corner has last value in col - just move it 2 cells right
    if (cr_last.col == dr_pen.col || cr_last.row == dr_pen.row ||
        (cr_last.col == dr_pen.col+1 && cr_last.row == dr_pen.row-1)) { // too close
      vertex_coord tmp_dr_last(dr_pen.row, dr_pen.col+2);
      append_path_to_res(cr_last, tmp_dr_last, false);
      cr_pen = vrtx.find_val_coord(r_pen_val);
    }
    append_path_to_res(cr_pen, dr_pen);
    cr_last = vrtx.find_val_coord(r_last_val);
    append_path_to_res(cr_last, dr_last);

    vertex_coord crz_dst(dr_pen.row-1, dr_pen.col);
    std::vector<vertex_coord> crz_path = bfs(lst_in_place, vrtx.zero_point(), crz_dst);
    std::vector<int> crz_moves = move_empty_cell(vrtx, crz_path);
    for (auto &m : crz_moves)
      res.push_back(m);
    res.push_back(vrtx.zero_move(VD_DOWN));
    res.push_back(vrtx.zero_move(VD_RIGHT));
    lst_in_place[dr_last.row][dr_last.col] = false;
//    std::cout << vrtx << std::endl;
    /********************************/
  }

  const vertex_move_direction move_pattern[] =
    {VD_UP, VD_RIGHT, VD_DOWN, VD_LEFT};
  int mpi = 0;
  while (!is_goal(vrtx)) {
    res.push_back(vrtx.zero_move(move_pattern[mpi]));
    ++mpi;
    mpi %= 4;
  }

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

#include "vertex.h"
#include <algorithm>
#include <cassert>
#include <exception>

int vertex::vertex_move_direction_to_delta_row(vertex_move_direction dir) const
{
  const int32_t delta_r[] = {-1, 0, 1, 0}; // WARNING! change according to the vertex_move_direction enum
  return delta_r[dir];
}

int vertex::vertex_move_direction_to_delta_col(vertex_move_direction dir) const
{
  const int32_t delta_c[] = {0, -1, 0, 1}; // WARNING! change according to the vertex_move_direction enum
  return delta_c[dir];
}
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

vertex::vertex(const state_t &state)
    : m_field_size(state.size()), m_state(state), m_cost(0), m_heuristics(0),
      m_parent(nullptr), m_zero(0,0) {
  for (int32_t r = 0; r < m_field_size; ++r) {
    for (int32_t c = 0; c < m_field_size; ++c) {
      if (m_state[r][c]) continue;
      m_zero.row = r;
      m_zero.col = c;
      break;
    }
  }
}
//////////////////////////////////////////////////////////////

vertex::vertex(const std::shared_ptr<vertex> parent)
    : m_field_size(parent->m_field_size), m_state(parent->m_state),
      m_cost(parent->m_cost + 1), m_heuristics(parent->m_heuristics),
      m_parent(parent), m_zero(parent->m_zero) {
}
//////////////////////////////////////////////////////////////

void vertex::recalculate_heuristics()
{
  m_heuristics = 0;
  for (int32_t r = 0; r < m_field_size; r++) {
    for (int32_t c = 0; c < m_field_size; c++) {
      if (m_state[r][c] <= 0)
        continue;

      m_heuristics +=
          abs(manhattan_heuristics::rows(m_field_size)[m_state[r][c]] - r) +
          abs(manhattan_heuristics::cols(m_field_size)[m_state[r][c]] - c);
    }
  }
}
//////////////////////////////////////////////////////////////

bool vertex::zero_move_is_possible(vertex_move_direction dir) const
{
  int32_t nr = m_zero.row + vertex_move_direction_to_delta_row(dir);
  int32_t nc = m_zero.col + vertex_move_direction_to_delta_col(dir);
  return !(nr < 0 || nc < 0 ||
           nr >= m_field_size || nc >= m_field_size);
}
//////////////////////////////////////////////////////////////

int32_t vertex::zero_move(vertex_move_direction dir)
{
  assert(zero_move_is_possible(dir));
  int32_t zr = m_zero.row;
  int32_t zc = m_zero.col;
  int32_t nr = zr + vertex_move_direction_to_delta_row(dir);
  int32_t nc = zc + vertex_move_direction_to_delta_col(dir);
  m_zero.row = nr;
  m_zero.col = nc;
  m_tile = m_state[nr][nc];
  std::swap(m_state[nr][nc], m_state[zr][zc]);
  return m_tile;
}
//////////////////////////////////////////////////////////////

vertex_coord vertex::find_val_coord(int32_t val) const
{
  for (int32_t r = 0; r < m_field_size; ++r) {
    for (int32_t c = 0; c < m_field_size; ++c) {
      if (m_state[r][c] != val) continue;
      return vertex_coord(r,c);
    }
  }
  throw new std::logic_error("value was not found in state");
}
//////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

static const int32_t MAX_FIELD_SIZE = 10;
std::vector<std::vector<int32_t>>
    manhattan_heuristics::dct_manhattan_rows(MAX_FIELD_SIZE);
std::vector<std::vector<int32_t>>
    manhattan_heuristics::dct_manhattan_cols(MAX_FIELD_SIZE);

void manhattan_heuristics::fill_rows_and_cols(int32_t n) {
  std::vector<int32_t> lst_rows(n * n);
  std::vector<int32_t> lst_cols(n * n);
  /*these two needs to be calculated for manhattan distance based heuristics*/
  for (int32_t r = 0; r < n; ++r) {
    for (int32_t c = 0; c < n; ++c) {
      lst_rows[r * n + c] = r;
      lst_cols[r * n + c] = c;
    }
  }
  // shift left
  std::rotate(lst_rows.rbegin(), lst_rows.rbegin() + 1, lst_rows.rend());
  std::rotate(lst_cols.rbegin(), lst_cols.rbegin() + 1, lst_cols.rend());
  dct_manhattan_rows[n] = lst_rows;
  dct_manhattan_cols[n] = lst_cols;
}
//////////////////////////////////////////////////////////////

const std::vector<int32_t> &manhattan_heuristics::rows(int32_t n) {
  assert(n >= 0 && n < MAX_FIELD_SIZE);
  if (dct_manhattan_rows[n].empty())
    fill_rows_and_cols(n);
  return dct_manhattan_rows[n];
}
//////////////////////////////////////////////////////////////

const std::vector<int32_t> &manhattan_heuristics::cols(int32_t n) {
  assert(n >= 0 && n < MAX_FIELD_SIZE);
  if (dct_manhattan_cols[n].empty())
    fill_rows_and_cols(n);
  return dct_manhattan_cols[n];
}
//////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& out, const vertex& v) {
  for (int32_t r = 0; r < v.m_field_size; ++r) {
    for (int32_t c = 0; c < v.m_field_size; ++c) {
      out << v.m_state[r][c] << "\t";
    }
    out << "\n";
  }
  return out;
}

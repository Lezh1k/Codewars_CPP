#include "vertex.h"
#include <algorithm>
#include <cassert>
#include <exception>
#include <stdexcept>

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
    : m_field_size(state.size()),
      m_state(state),
      m_zero(0,0) {
  m_zero = find_val_coord(0);
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
  std::swap(m_state[nr][nc], m_state[zr][zc]);
  return m_state[zr][zc];
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

std::ostream& operator<<(std::ostream& out, const vertex& v) {
  for (int32_t r = 0; r < v.m_field_size; ++r) {
    for (int32_t c = 0; c < v.m_field_size; ++c) {
      out << v.m_state[r][c] << "\t";
    }
    out << "\n";
  }
  return out;
}
//////////////////////////////////////////////////////////////

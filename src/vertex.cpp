#include "vertex.h"
#include <algorithm>
#include <cassert>

vertex::vertex(const state_t &state)
    : m_field_size(state.size()), m_state(state), m_cost(0), m_heuristics(0),
      m_parent(nullptr) {}

vertex::vertex(const std::shared_ptr<vertex> parent)
    : m_field_size(parent->m_field_size), m_state(parent->m_state),
      m_cost(parent->m_cost + 1), m_heuristics(parent->m_heuristics),
      m_parent(parent) {}
//////////////////////////////////////////////////////////////

void vertex::recalculate_heuristics() {
  m_heuristics = 0;
  for (int r = 0; r < m_field_size; r++) {
    for (int c = 0; c < m_field_size; c++) {
      if (!m_state[r][c])
        continue;

      m_heuristics +=
          abs(manhattan_heuristics::rows(m_field_size)[m_state[r][c]] - r) +
          abs(manhattan_heuristics::cols(m_field_size)[m_state[r][c]] - c);
    }
  }
}
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

static const int MAX_FIELD_SIZE = 10;
std::vector<std::vector<int>>
    manhattan_heuristics::dct_manhattan_rows(MAX_FIELD_SIZE);
std::vector<std::vector<int>>
    manhattan_heuristics::dct_manhattan_cols(MAX_FIELD_SIZE);

void manhattan_heuristics::fill_rows_and_cols(int n) {
  std::vector<int> lst_rows(n * n);
  std::vector<int> lst_cols(n * n);
  /*these two needs to be calculated for manhattan distance based heuristics*/
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
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

const std::vector<int> &manhattan_heuristics::rows(int n) {
  assert(n >= 0 && n < MAX_FIELD_SIZE);
  if (dct_manhattan_rows[n].empty())
    fill_rows_and_cols(n);
  return dct_manhattan_rows[n];
}
//////////////////////////////////////////////////////////////

const std::vector<int> &manhattan_heuristics::cols(int n) {
  assert(n >= 0 && n < MAX_FIELD_SIZE);
  if (dct_manhattan_cols[n].empty())
    fill_rows_and_cols(n);
  return dct_manhattan_cols[n];
}
//////////////////////////////////////////////////////////////

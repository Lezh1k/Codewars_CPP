#include "queens.h"
#include <algorithm>

#include <iostream>
#include <iomanip>

const int chess_board::NOT_SET = -1;
chess_board::chess_board(int N) :
  m_field_size(N),
  m_init_queen_row(-1),
  m_init_queen_col(-1),
  m_board(N*N),
  m_cols(N)
{
  std::fill(m_cols.begin(), m_cols.end(), chess_board::NOT_SET);
  std::vector<int> rows_mc;
  for (int c = 0; c < N; ++c) {
    rows_with_min_conflicts(c, rows_mc);
    int rnd = std::rand() % rows_mc.size();
    set_queen_state(rows_mc[rnd], c, true);
    rows_mc.erase(rows_mc.begin(), rows_mc.end());
  }
}

chess_board::chess_board(int N,
                         int initial_row,
                         int initial_col)  :
  m_field_size(N),
  m_init_queen_row(initial_row),
  m_init_queen_col(initial_col),
  m_board(N*N),
  m_cols(N)
{
  std::fill(m_cols.begin(), m_cols.end(), chess_board::NOT_SET);
  set_queen_state(initial_row, initial_col, true);
  std::vector<int> rows_mc;
  rows_mc.reserve(m_field_size);
  for (int c = 0; c < N; ++c) {
    if (c == initial_col)
      continue;
    rows_with_min_conflicts(c, rows_mc);
    int rnd = std::rand() % rows_mc.size();
    set_queen_state(rows_mc[rnd], c, true);
    rows_mc.erase(rows_mc.begin(), rows_mc.end());
  }
}
//////////////////////////////////////////////////////////////

void chess_board::set_queen_state(int row, int col, bool state)
{
  m_board[row*m_field_size + col] = state;
  m_cols[col] = state ? row : NOT_SET;
}
//////////////////////////////////////////////////////////////

void chess_board::rows_with_min_conflicts(int col, std::vector<int> &res) const
{  
  res.reserve(m_field_size); //max
  int min = m_field_size+1;
  for (int r = 0; r < m_field_size; ++r) {
    int conflicts = conflicts_count(r, col);
    if (conflicts > min) {
      continue;
    } else if (conflicts == min) {
      res.push_back(r);
    } else {
      res.erase(res.begin(), res.end());
      min = conflicts;
      res.push_back(r);
    }
  }
}
//////////////////////////////////////////////////////////////

bool chess_board::find_solution(int max_depth)
{
  std::vector<int> cols; //they can be changed
  cols.reserve(m_field_size-1);
  for (int i = 0; i < m_field_size; ++i) {
    if (i == m_init_queen_col)
      continue;
    cols.push_back(i);
  }

  std::vector<int> possible_rows;
  possible_rows.reserve(m_field_size);

  std::vector<int> possible_cols;
  possible_cols.reserve(m_field_size-1);

  while (max_depth--) {
    if (is_goal_state()) {
      std::cout << max_depth << std::endl;
      return true;
    }

    possible_cols.erase(possible_cols.begin(), possible_cols.end());
    for (int c = 0; c < m_field_size; ++c) {
      if (c == m_init_queen_col) continue;
      int cc = conflicts_count(m_cols[c], c);
      if (cc == 0) continue;
      possible_cols.push_back(c);
    }

    int rnd_col_idx = std::rand() % possible_cols.size();
    int rnd_col = possible_cols[rnd_col_idx];

    rows_with_min_conflicts(rnd_col, possible_rows);
    int rnd_row_idx = std::rand() % possible_rows.size();
    int rnd_row = possible_rows[rnd_row_idx];

    set_queen_state(m_cols[rnd_col], rnd_col, false);
    set_queen_state(rnd_row, rnd_col, true);
    possible_rows.erase(possible_rows.begin(), possible_rows.end());
  }
  return false;
}
//////////////////////////////////////////////////////////////

bool chess_board::is_goal_state() const
{
  for (int c = 0; c < m_field_size; ++c) {
    if (conflicts_count(m_cols[c], c))
      return false;
  }
  return true;
}
//////////////////////////////////////////////////////////////

int chess_board::conflicts_count(int row, int col) const
{
  int count = 0;
  int d1 = row + col;
  int d2 = row - col + m_field_size;

  for (int c = 0; c < m_field_size; ++c) {
    if (c == col)
      continue;
    if (m_cols[c] == NOT_SET)
      continue;
    int r = m_cols[c];
    if (r == row ||
        r + c == d1 ||
        r - c + m_field_size == d2) {
      ++count;
    }
  }
  return count;
}
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////


std::ostream& operator<<(std::ostream& out, const chess_board& v) {
  for (int32_t r = 0; r < v.m_field_size; ++r) {
    for (int32_t c = 0; c < v.m_field_size; ++c) {
      int cc = v.conflicts_count(r,c);
      out << std::setw(3) <<
             (v.m_board[r*v.m_field_size + c] ? 'Q' : '.') << cc << " ";
    }
    out << "\n";
  }
  return out;
}
//////////////////////////////////////////////////////////////


std::string chess_board::to_string() const
{
  std::string res;
  res.reserve(m_board.size() + m_field_size);
  for (int r = 0; r < m_field_size; ++r) {
    for (int c = 0; c < m_field_size; ++c) {
      res.push_back(m_board[r * m_field_size + c] ? 'Q' : '.');
    }
    res.push_back('\n');
  }
  return res;
}
//////////////////////////////////////////////////////////////

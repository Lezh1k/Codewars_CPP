#include "queens.h"
#include <iostream>

chess_board::chess_board(int N) :
  m_field_size(N),
  m_init_queen_row(-1),
  m_init_queen_col(-1),
  m_board(N*N),
  m_used_rows(N),
  m_used_cols(N),
  m_used_diag_slash(2*N),
  m_used_diag_backslash(2*N),
  m_cols(N) {
}

chess_board::chess_board(int N,
                         int initial_row,
                         int initial_col)  :
  m_field_size(N),
  m_init_queen_row(initial_row),
  m_init_queen_col(initial_col),
  m_board(N*N),
  m_used_rows(N),
  m_used_cols(N),
  m_used_diag_slash(2*N),
  m_used_diag_backslash(2*N),
  m_cols(N)
{
  set_queen_state(m_init_queen_row, m_init_queen_col, true);
}

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

bool chess_board::find_solution_backtrack()
{
  return backtracking_bruteforce(0);
}
//////////////////////////////////////////////////////////////

void chess_board::set_queen_state(int row,
                                  int col,
                                  bool is_set)
{
  m_board[row * m_field_size + col] = is_set;
  m_used_rows[row] = is_set;
  m_used_cols[col] = is_set;
  m_used_diag_slash[col + row] = is_set;
  m_used_diag_backslash[row - col + m_field_size] = is_set;

}
//////////////////////////////////////////////////////////////


bool chess_board::coordinate_is_safe(int row,
                                     int col) const
{
  return !m_used_rows[row] &&
      !m_used_cols[col] &&
      !m_used_diag_slash[row + col] &&
      !m_used_diag_backslash[row - col + m_field_size];
}
//////////////////////////////////////////////////////////////

bool chess_board::backtracking_bruteforce(int col)
{
  if (col >= m_field_size)
    return true;

  if (col == m_init_queen_col)
    return backtracking_bruteforce(col+1);

  for (int r = 0; r < m_field_size; ++r) {
    if (!coordinate_is_safe(r, col))
      continue;

    set_queen_state(r, col, true);
    if (backtracking_bruteforce(col+1))
      return true;
    set_queen_state(r, col, false);
  }
  return false;
}
//////////////////////////////////////////////////////////////

int chess_board::attacks_count(int row, int col) const
{
  int count = 0;
  int d1 = row + col;
  int d2 = row - col + m_field_size;

  for (int c = 0; c < m_field_size; ++c) {
    if (c == col)
      continue;

    for (int r = 0; r < m_field_size; ++r) {

    }
  }
  return count;
}
//////////////////////////////////////////////////////////////

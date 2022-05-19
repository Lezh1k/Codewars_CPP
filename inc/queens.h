#ifndef QUEENS_H
#define QUEENS_H

#include <vector>
#include <string>


class chess_board {
private:
  int m_field_size;
  int m_init_queen_row;
  int m_init_queen_col;
  std::vector<bool> m_board;

  /*backtracking*/
  std::vector<bool> m_used_rows;
  std::vector<bool> m_used_cols;
  std::vector<bool> m_used_diag_slash;
  std::vector<bool> m_used_diag_backslash;
  void set_queen_state(int row, int col, bool is_set);
  bool coordinate_is_safe(int row, int col) const;
  bool backtracking_bruteforce(int col);

  /*min conflicts*/
  static const int VAL_IN_COL_IS_NOT_SET = -1;
  std::vector<int> m_cols;
  int attacks_count(int row, int col) const;

public:
  explicit chess_board(int N);
  chess_board(int N, int initial_row, int initial_col);

  std::string to_string() const;
  bool find_solution_backtrack();
};

#endif // QUEENS_H

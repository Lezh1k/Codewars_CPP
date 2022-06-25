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

  static const int NOT_SET;
  std::vector<int> m_cols;
  void set_queen_state(int row, int col, bool state);

  void rows_with_min_conflicts(int col, std::vector<int> &res) const;
  int conflicts_count(int row, int col) const;
  bool is_goal_state() const;

public:
  explicit chess_board(int N);
  chess_board(int N, int initial_row, int initial_col);

  std::string to_string() const;
  bool find_solution(int max_depth = 10000);

  //debug!
  friend std::ostream& operator<<(std::ostream& out, const chess_board& v);
};

#endif // QUEENS_H

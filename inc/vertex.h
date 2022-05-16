#ifndef _C_VERTEX_H
#define _C_VERTEX_H

#include <cstdint>
#include <memory>
#include <vector>

typedef std::vector<std::vector<int>> state_t;

//////////////////////////////////////////////////////////////

struct vertex_coord {
  int row, col;
  vertex_coord() : row(0), col(0) {}
  vertex_coord(int r, int c) : row(r), col(c){}
};

inline bool operator==(const vertex_coord& lhs, const vertex_coord& rhs){
  return lhs.row == rhs.row && lhs.col == rhs.col;
}
inline bool operator!=(const vertex_coord& lhs, const vertex_coord& rhs){ return !(lhs == rhs); }
//////////////////////////////////////////////////////////////

enum vertex_move_direction {
  VD_UP = 0,
  VD_LEFT,
  VD_DOWN,
  VD_RIGHT
};

class vertex {
private:
  const int m_field_size;
  state_t m_state; 
  vertex_coord m_zero;

  inline int vertex_move_direction_to_delta_row(vertex_move_direction dir) const;
  inline int vertex_move_direction_to_delta_col(vertex_move_direction dir) const;
public:
  vertex() = delete;
  explicit vertex(const state_t &state);
  vertex(const vertex &v) = delete;
  ~vertex() = default;

  const state_t &state() const { return m_state; } 
  const vertex_coord& zero_point() const { return m_zero; }

  // operators
  const int &operator()(size_t row, size_t col) const {
    return m_state[row][col];
  }
  int &operator()(size_t row, size_t col) { return m_state[row][col]; }
  const int &operator()(const vertex_coord &c) const {
    return (*this)(c.row, c.col);
  }
  int &operator()(const vertex_coord &c) { return (*this)(c.row,c.col); }
  // methods
  bool zero_move_is_possible(vertex_move_direction dir) const;
  int32_t zero_move(vertex_move_direction dir);
  vertex_coord find_val_coord(int32_t val) const;
  friend std::ostream& operator<<(std::ostream& out, const vertex& v);
};

#endif

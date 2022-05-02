#ifndef _C_VERTEX_H
#define _C_VERTEX_H

#include <cstdint>
#include <vector>

typedef std::vector<std::vector<int>> state_t;

class manhattan_heuristics {
private:
  static std::vector<std::vector<int>> dct_manhattan_rows;
  static std::vector<std::vector<int>> dct_manhattan_cols;
  static void fill_rows_and_cols(int n);

public:
  manhattan_heuristics() = delete;
  ~manhattan_heuristics() = delete;

  static const std::vector<int> &rows(int n);
  static const std::vector<int> &cols(int n);
};
//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

class vertex
{
private:
  const int m_field_size;
  state_t m_state;
  int m_cost;
  int m_heuristics;
  const vertex *m_parent;
  int m_tile;

public:
  vertex() = delete;
  explicit vertex(const state_t &state);
  vertex(const vertex *parent);
  vertex(const vertex &v) = delete;
  ~vertex() = default;

  const state_t &state() const {return m_state;}  
  int cost() const {return m_cost;}
  int heuristics() const {return m_heuristics;}
  const vertex* parent() const {return m_parent;}

  int tile() const {return m_tile;}
  void set_tile(int tile) {m_tile = tile;}

  //operators
  const int &operator()(size_t row, size_t col) const {return m_state[row][col];}
  int &operator()(size_t row, size_t col) {return m_state[row][col];}
  // methods
  void recalculate_heuristics();
};

#endif

#ifndef _GAME_MODEL_H
#define _GAME_MODEL_H

#include "vertex.h"

class Game {
private:
  int32_t m_field_size;
  state_t m_start_state;
  state_t m_goal_state;

  bool has_solution() const;
  bool is_goal(const vertex &game_state) const;

  std::vector<vertex_coord> bfs(const std::vector<std::vector<bool> > &lst_in_place,
                                const vertex_coord &src,
                                const vertex_coord &dst);

  std::vector<int32_t> move_not_empty_cell(vertex &vrtx,
                                           std::vector<std::vector<bool> > &lst_in_place,
                                           const vertex_coord &src,
                                           const vertex_coord &dst);

  std::vector<int32_t> move_empty_cell(vertex &vrtx,
                                       const std::vector<vertex_coord> &path);

public:
  Game() = delete;
  explicit Game(const std::vector<std::vector<int32_t>> &start_state);
  ~Game() = default;
  std::vector<int32_t> find_solution_by_strategy();
};

#endif

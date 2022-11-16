#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <vector>

enum direction_t { D_RIGHT = 0, D_DOWN, D_LEFT, D_UP };
static const direction_t DIRS[4] = {D_RIGHT, D_DOWN, D_LEFT, D_UP};
static const int8_t DELTA_ROW[4] = {0, 1, 0, -1};
static const int8_t DELTA_COL[4] = {1, 0, -1, 0};


struct coord_t {
  size_t row;
  size_t col;

  coord_t(): row(-1), col(-1) {}
  coord_t(size_t r, size_t c) : row(r), col(c) {}
};

inline bool operator==(const coord_t &l, const coord_t &r) {
  return l.row == r.row && l.col == r.col;
}

inline bool operator!=(const coord_t &l, const coord_t &r) {
  return !(l == r);
}

inline bool operator<(const coord_t &l, const coord_t &r) {
  if (l.row == r.row)
    return l.col < r.col;
  return l.row < r.row;
}

inline bool operator>(const coord_t &l, const coord_t &r) {
  return r < l;
}

inline bool operator<=(const coord_t &l, const coord_t &r) {
  return !(l > r);
}

inline bool operator>=(const coord_t &l, const coord_t &r) {
  return !(l < r);
}
//////////////////////////////////////////////////////////////

std::vector<std::string> lines_split(const std::string &to_split,
                                     const std::string &sep);
std::string lines_join(const std::vector<std::string> &lines,
                       const std::string &sep);

std::vector<std::string>
field_expand(const std::vector<std::string> &field);

std::vector<std::string> field_shrink(const std::vector<std::string> &field);

std::set<coord_t> fill_contour_return_border(std::vector<std::string> &field,
                                size_t row,
                                size_t col,
                                char sym);

std::vector<std::set<coord_t> > mark_contours(std::vector<std::string> &field);
std::set<coord_t> shrink_contour(const std::set<coord_t> &contour);

std::string contour_to_string(const std::vector<std::string> &field,
                              const std::set<coord_t> contour);

//////////////////////////////////////////////////////////////

std::vector<std::string> break_piece(const std::string &shape) {
  std::vector<std::string> field = lines_split(shape, "\n");
  std::vector<std::string> ex_field = field_expand(field);
  std::vector<std::set<coord_t>> contours = mark_contours(ex_field);

  std::vector<std::string> result;
  result.reserve(contours.size());
  for (const auto &contour : contours) {
    std::string str = contour_to_string(field, contour);
    result.push_back(str);
  }
  return result;
}
//////////////////////////////////////////////////////////////

std::vector<std::string> lines_split(const std::string &to_split,
                                          const std::string &sep) {
  std::vector<std::string> lst_res;
  size_t start = 0;
  while (start != std::string::npos) {
    size_t sep_idx = to_split.find_first_of(sep, start);
    if (sep_idx == std::string::npos) {
      lst_res.emplace_back(to_split.begin() + start, to_split.end());
      break;
    }
    lst_res.emplace_back(to_split.begin() + start, to_split.begin() + sep_idx);
    start = to_split.find_first_not_of(sep, sep_idx);
  };

  return lst_res;
}
//////////////////////////////////////////////////////////////

std::string lines_join(const std::vector<std::string> &lines,
                       const std::string &sep) {
  size_t res_len = std::accumulate(lines.begin(), lines.end(), lines.size() * sep.size(),
                  [](size_t acc, const std::string &x) { return x.size() + acc;});
  std::string res_str;
  res_str.reserve(res_len);
  for (auto &l : lines) {
    res_str.append(l);
    res_str.append(sep);
  }
  return res_str;
}
//////////////////////////////////////////////////////////////

std::vector<std::string>
field_expand(const std::vector<std::string> &field) {
  static const char POSSIBLE_CHARS_FOR_PLUS[4] = {'-', '|', '-', '|'};
  static const std::map<char, std::array<std::string, 3>> str_complements = {
      {' ', {"   ",
             "   ",
             "   "}},
      {'-', {"   ",
             "---",
             "   "}},
      {'|', {" | ",
             " | ",
             " | "}},
      {'+', {" | ",
             "-+-",
             " | "}},
  };

  std::vector<std::string> res(field.size() * 3);
  for (size_t r = 0; r < field.size(); ++r) {
    res[r * 3 + 0].resize(field[r].size() * 3);
    res[r * 3 + 1].resize(field[r].size() * 3);
    res[r * 3 + 2].resize(field[r].size() * 3);
  }

  for (size_t r = 0; r < field.size(); ++r) {
    for (size_t c = 0; c < field[r].size(); ++c) {
      const auto it = str_complements.find(field[r][c]);
      for (int i = 0; i < 3; ++i) {
        res[r * 3 + i].replace(c * 3, 3, it->second[i]);
      }

      if (it->first != '+')
        continue; // no need extra handling

      for (auto dir : DIRS) {
        size_t nr = r + DELTA_ROW[dir];
        size_t nc = c + DELTA_COL[dir];

        if (nr < field.size() && nc < field[r].size() &&
            (field[nr][nc] == '+' ||
             field[nr][nc] == POSSIBLE_CHARS_FOR_PLUS[dir])) {
          continue; // no need to remove extra symbols.
        }

        // take center of extension block. and remove additional symbol on side
        // = dir
        size_t enr = r * 3 + 1 + DELTA_ROW[dir];
        size_t enc = c * 3 + 1 + DELTA_COL[dir];
        res[enr][enc] = ' ';
      } // for dir : DIRS
    }   // for cols
  }     // for rows in field
  return res;
}
//////////////////////////////////////////////////////////////

// fills contour and returns borders
std::set<coord_t>
fill_contour_return_border(std::vector<std::string> &field,
                           size_t row,
                           size_t col,
                           char sym) {
  std::set<coord_t> res_set;
  std::queue<coord_t> q;
  q.push(coord_t(row, col));

  while (!q.empty()) {
    coord_t vert = q.front();
    q.pop();

    if (field[vert.row][vert.col] != ' ')
      continue; // already visited or marked

    field[vert.row][vert.col] = sym;
    for (int dr = -1; dr <= 1; ++dr) {
      size_t nr = vert.row + dr;
      if (nr >= field.size())
        continue; // out of bounds;

      for (int dc = -1; dc <= 1; ++dc) {
        size_t nc = vert.col + dc;
        if (nc >= field[nr].size())
          continue; // out of bounds

        if (field[nr][nc] == sym)
          continue; // already marked

        if (field[nr][nc] != ' ') {
          res_set.insert(coord_t(nr, nc));
          continue; // border
        }
        q.push(coord_t(nr, nc));
      } // for dc
    }   // for dr
  }     // while (!q.empty())
  return res_set;
}
//////////////////////////////////////////////////////////////

std::vector<std::set<coord_t>>
mark_contours(std::vector<std::string> &field) {
  char cm = '0';
  fill_contour_return_border(field, 0, 0, cm++);
  std::vector<std::set<coord_t>> res;
  for (size_t r = 0; r < field.size(); ++r) {
    for (size_t c = 0; c < field[r].size(); ++c) {
      if (field[r][c] != ' ')
        continue;

      std::set<coord_t> contour = fill_contour_return_border(field, r, c, cm++);
      std::set<coord_t> shrinked = shrink_contour(contour);
      res.push_back(shrinked);
    } // for cols
  }   // for rows
  return res;
}
//////////////////////////////////////////////////////////////

std::set<coord_t> shrink_contour(const std::set<coord_t> &contour) {
  std::set<coord_t> res;
  for (const auto &c : contour) {
    coord_t nit(c.row / 3, c.col / 3);
    res.insert(nit);
  }
  return res;
}
//////////////////////////////////////////////////////////////

std::vector<std::string>
contour_to_rows(const std::vector<std::string> &field,
                const std::set<coord_t> contour) {
  std::vector<std::string> rows;
  std::string line;
  size_t prev_row = contour.begin()->row;
  int prev_col = -1;
  size_t min_col = contour.begin()->col;

  for (auto curr_it = contour.cbegin(); curr_it != contour.cend(); ++curr_it) {
    min_col = std::min(curr_it->col, min_col);

    if (curr_it->row != prev_row) {
      rows.push_back(line);
      line.clear();
      prev_col = -1;
      prev_row = curr_it->row;
    }

    size_t extra_str_len = curr_it->col - prev_col;
    if (extra_str_len > 1) {
      std::string extra_str(extra_str_len-1, ' ');
      line.append(extra_str);
    }

    line.push_back(field[curr_it->row][curr_it->col]);
    prev_col = curr_it->col;
  }

  if (!line.empty())
    rows.push_back(line);

  if (min_col > 0) {
    for (auto &r : rows) {
      r.erase(r.begin(), r.begin()+min_col);
    }
  }

  return rows;
}
//////////////////////////////////////////////////////////////

std::string
contour_to_string(const std::vector<std::string> &field,
                  const std::set<coord_t> contour) {
  std::vector<std::string> rows = contour_to_rows(field, contour);

  for (size_t r = 0; r < rows.size(); ++r) {
    for (size_t c = 0; c < rows[r].size(); ++c) {
      if (rows[r][c] != '+')
        continue;

      // check left and right neighbours
      size_t lc = c + DELTA_COL[D_LEFT];
      size_t rc = c + DELTA_COL[D_RIGHT];
      if (lc < rows[r].size() && rc < rows[r].size()) {
        if ( (rows[r][lc] == '-' || rows[r][lc] == '+') &&
             (rows[r][rc] == '-' || rows[r][rc] == '+') ) {
          rows[r][c] = '-';
        }
      }

      // check top and bottom neighbours
      size_t tr = r + DELTA_ROW[D_UP];
      size_t br = r + DELTA_ROW[D_DOWN];
      if (tr < rows.size() && br < rows.size()) {
        if ( (rows[tr][c] == '|' || rows[tr][c] == '+') &&
             (rows[br][c] == '|' || rows[br][c] == '+') ) {
          rows[r][c] = '|';
        }
      }

    } // for (size_t c = 0; c < rows[r].size(); ++c)
  } // for (size_t r = 0; r < rows.size(); ++r)

  return lines_join(rows, "\n");
}
//////////////////////////////////////////////////////////////

void bp_main()
{
  std::string tst_shapes[] = {
    "+------------+\n"
    "|      +-----+\n"
    "|      |     |\n"
    "|      |     |\n"
    "+---+--+-----+\n"
    "|   |        |\n"
    "|   |        |\n"
    "+---+--+-----+\n"
    "|      |     |\n"
    "|      +-----+\n"
    "+------------+",

    "    +---+\n"
    "    |   |\n"
    "+---+   |\n"
    "|       |\n"
    "+-------+",

    "+-----------------+\n"
    "|                 |\n"
    "|   +-------------+\n"
    "|   |\n"
    "|   |\n"
    "|   |\n"
    "|   +-------------+\n"
    "|                 |\n"
    "|                 |\n"
    "+-----------------+",
  };

  for (auto &shape : tst_shapes) {
     std::vector<std::string> pieces = break_piece(shape);
     for (auto &p : pieces) {
       std::cout << p << "\n***\n";
     }

     std::cout << "*********************" << std::endl;
  }
}

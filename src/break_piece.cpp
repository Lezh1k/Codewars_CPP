#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <stack>
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

static const coord_t COORD_NOT_INITIALIZED;
//////////////////////////////////////////////////////////////

std::vector<std::vector<uint16_t>>
lines_split(const std::string &shape,
            const std::string &sep);

std::string
lines_join(const std::vector<std::vector<uint16_t>> &lines,
           const std::string &sep);

std::vector<std::vector<uint16_t> >
field_expand(const std::vector<std::vector<uint16_t> > &field);

void ff_fill(std::vector<std::vector<uint16_t>> &field,
             size_t row,
             size_t col,
             uint16_t sym);

void
ff_scan(const std::vector<std::vector<uint16_t>> &field,
        size_t lx,
        size_t rx,
        size_t y,
        std::stack<coord_t> &s);

std::vector<std::set<coord_t> >
mark_contours(std::vector<std::vector<uint16_t> > &field);

std::set<coord_t>
shrink_contour(const std::set<coord_t> &contour);

std::array<coord_t, 4> get_valid_neigbours(const std::vector<std::vector<uint16_t>> &rows, size_t row, size_t col,
                                           size_t &out_neigh_number);
void
fix_contour_borders(std::vector<std::vector<uint16_t>> &rows);

std::vector<std::vector<uint16_t>>
shrink_rows(const std::vector<std::vector<uint16_t>>& rows);

std::vector<std::vector<uint16_t>>
contour_to_rows(const std::vector<std::vector<uint16_t>> &ex_field,
                const std::set<coord_t> &contour);

std::string
contour_to_string(const std::vector<std::vector<uint16_t>> &ex_field,
                  const std::set<coord_t> &contour);

//////////////////////////////////////////////////////////////

std::vector<std::string>
break_piece(const std::string &shape) {
  std::vector<std::vector<uint16_t>> field = lines_split(shape, "\n");
  std::cout << lines_join(field, "\n") << std::endl;

  std::vector<std::vector<uint16_t>> ex_field = field_expand(field);
  std::cout << lines_join(ex_field, "\n") << std::endl;

  std::vector<std::set<coord_t>> contours = mark_contours(ex_field);
  std::vector<std::string> result;
  result.reserve(contours.size());
  for (const auto &contour : contours) {
    std::string str = contour_to_string(ex_field, contour);
    result.push_back(str);
  }
  return result;
}
//////////////////////////////////////////////////////////////

std::vector<std::vector<uint16_t> >
lines_split(const std::string &shape,
            const std::string &sep) {
  std::vector<std::vector<uint16_t> > lst_res;
  size_t start = 0;
  while (start != std::string::npos) {
    size_t sep_idx = shape.find_first_of(sep, start);
    if (sep_idx == std::string::npos) {
      std::vector<uint16_t> row;
      size_t row_len = std::distance(shape.begin() + start, shape.end());
      row.reserve(row_len);
      std::copy(shape.begin() + start, shape.end(), std::back_inserter(row));
      lst_res.push_back(row);
      break;
    }
    std::vector<uint16_t> row;
    size_t row_len = sep_idx - start;
    row.reserve(row_len);
    std::copy(shape.begin() + start, shape.begin() + sep_idx, std::back_inserter(row));
    lst_res.push_back(row);
    start = shape.find_first_not_of(sep, sep_idx);
  };

  return lst_res;
}
//////////////////////////////////////////////////////////////

std::string
lines_join(const std::vector<std::vector<uint16_t> > &lines,
           const std::string &sep) {
  size_t res_len = std::accumulate(lines.begin(), lines.end(), lines.size() * sep.size(),
                                   [](size_t acc, const std::vector<uint16_t> &x) { return x.size() + acc;});
  std::string res_str;
  res_str.reserve(res_len);

  for (auto &l : lines) {
    std::copy(l.begin(), l.end(), std::back_inserter(res_str));
    res_str.append(sep);
  }
  return res_str;
}
//////////////////////////////////////////////////////////////

std::vector<std::vector<uint16_t>>
field_expand(const std::vector<std::vector<uint16_t>> &field) {
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

  std::vector<std::vector<uint16_t>> res(field.size() * 3);
  for (size_t r = 0; r < field.size(); ++r) {
    res[r * 3 + 0].resize(field[r].size() * 3);
    res[r * 3 + 1].resize(field[r].size() * 3);
    res[r * 3 + 2].resize(field[r].size() * 3);
  }

  for (size_t r = 0; r < field.size(); ++r) {
    for (size_t c = 0; c < field[r].size(); ++c) {
      const auto it = str_complements.find(field[r][c]);
      for (int i = 0; i < 3; ++i) {
        std::transform(it->second[i].begin(),
                       it->second[i].end(),
                       res[r*3+i].begin() + c*3,
            [](char in) { return in; });
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

void ff_scan(const std::vector<std::vector<uint16_t> > &field,
             size_t lx,
             size_t rx,
             size_t y,
             std::stack<coord_t> &s) {
  if (y >= field.size())
    return; //out of bounds;

  bool added = false;
  while (static_cast<int32_t>(lx) < 0)
    ++lx;

  for (size_t x = lx; x < field[y].size() && x <= rx; ++x) {
    if (field[y][x] != ' ')
      added = false;
    else if (!added) {
      s.push(coord_t(y, x));
      added = true;
    }
  }
}
//////////////////////////////////////////////////////////////

void
ff_fill(std::vector<std::vector<uint16_t> > &field,
        size_t row,
        size_t col,
        uint16_t sym) {
  std::stack<coord_t> s;
  s.push(coord_t(row, col));

  while (!s.empty()) {
    coord_t t = s.top();
    s.pop();
    size_t lx = t.col - 1;
    while ( lx < field[t.row].size() && field[t.row][lx] == ' ' ) {
      field[t.row][lx--] = sym;
    }
    size_t rx = t.col; // including t.col
    while ( rx < field[t.row].size() && field[t.row][rx] == ' ') {
      field[t.row][rx++] = sym;
    }
    ff_scan(field, lx, rx, t.row - 1, s);
    ff_scan(field, lx, rx, t.row + 1, s);
  }
}
//////////////////////////////////////////////////////////////

std::vector<std::set<coord_t>>
mark_contours(std::vector<std::vector<uint16_t>> &field) {
  uint16_t cm = 1000;
  ff_fill(field, 0, 0, cm-1);
  std::vector<std::set<coord_t>> res;
  for (size_t r = 0; r < field.size(); ++r) {
    for (size_t c = 0; c < field[r].size(); ++c) {
      if (field[r][c] != ' ')
        continue;
      ff_fill(field, r, c, cm++);
    } // for cols
  }   // for rows
  res.resize(cm-1000);


  for (size_t r = 0; r < field.size(); ++r) {
    for (size_t c = 0; c < field[r].size(); ++c) {
      if (field[r][c] >= 999)
        continue; // it's mark

      for (int dr = -1; dr <= 1; ++dr) {
        size_t nr = r + dr;
        if (nr >= field.size())
          continue; //out of bounds

        for (int dc = -1; dc <= 1; ++dc) {
          size_t nc = c + dc;
          if (nc >= field[nr].size())
            continue; //out of bounds

          if (field[nr][nc] <= 999)
            continue; // it's another part of border. we need marks around current symbol

          res[field[nr][nc] - 1000].insert(coord_t(r,c));
        } // for dc
      } // for dr
    } // for cols
  } // for rows


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

std::array<coord_t, 4>
get_valid_neigbours(const std::vector<std::vector<uint16_t>> &rows,
                    size_t row,
                    size_t col,
                    size_t &out_neigh_number) {
  static const char possible_chars_for_plus[] = {'-', '|', '-', '|'};
  std::array<coord_t, 4> valid_neighbours;
  out_neigh_number = 0;
  for (auto dir : DIRS) {
    size_t nr = row + DELTA_ROW[dir];
    size_t nc = col + DELTA_COL[dir];

    if (nr >= rows.size())
      continue;

    if (nc >= rows[nr].size())
      continue;

    if (rows[nr][nc] != possible_chars_for_plus[dir] && rows[nr][nc] != '+')
      continue;

    valid_neighbours[dir] = coord_t(nr, nc);
    ++out_neigh_number;
  }
  return valid_neighbours;
}
//////////////////////////////////////////////////////////////

void
fix_contour_borders(std::vector<std::vector<uint16_t>> &rows) {
  for (size_t r = 0; r < rows.size(); ++r) {
    for (size_t c = 0; c < rows[r].size(); ++c) {
      if (rows[r][c] == ' ')
        continue;

      size_t valid_neigbours_count = 0;
      std::array<coord_t, 4> valid_neighbours =
          get_valid_neigbours(rows, r, c, valid_neigbours_count);

      if (valid_neigbours_count > 2)
        continue; // this is some intersection

      if (valid_neighbours[D_LEFT] != COORD_NOT_INITIALIZED &&
          valid_neighbours[D_RIGHT] != COORD_NOT_INITIALIZED) {
        rows[r][c] = '-';
      }

      if (valid_neighbours[D_UP] != COORD_NOT_INITIALIZED &&
          valid_neighbours[D_DOWN] != COORD_NOT_INITIALIZED) {
        rows[r][c] = '|';
      }
    } // for cols
  } // for rows
}
//////////////////////////////////////////////////////////////

std::vector<std::vector<uint16_t>>
shrink_rows(const std::vector<std::vector<uint16_t>>& rows) {
  std::vector<std::vector<uint16_t>> res;
  for (size_t r = 0; r < rows.size(); r += 3) {
    std::vector<uint16_t> nrow;
    nrow.reserve(rows[r].size() / 3 + 1);
    for (size_t c = 0; c < rows[r].size(); c += 3) {
      nrow.push_back(rows[r][c]);
    }
    res.push_back(nrow);
  }
  return res;
}
//////////////////////////////////////////////////////////////

std::vector<std::vector<uint16_t>>
contour_to_rows(const std::vector<std::vector<uint16_t>> &ex_field,
                const std::set<coord_t> &contour) {
  std::vector<std::vector<uint16_t>> rows;
  std::vector<uint16_t> line;
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
      std::copy(extra_str.begin(), extra_str.end(), std::back_inserter(line));
    }

    line.push_back(static_cast<char>(ex_field[curr_it->row][curr_it->col]));
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
contour_to_string(const std::vector<std::vector<uint16_t> > &ex_field,
                  const std::set<coord_t> &contour) {
  std::vector<std::vector<uint16_t> > rows = contour_to_rows(ex_field, contour);
  fix_contour_borders(rows);
  std::vector<std::vector<uint16_t> > shrinked_rows = shrink_rows(rows);
  return lines_join(shrinked_rows, "\n");
}
//////////////////////////////////////////////////////////////

void bp_main()
{
  std::string tst_shapes[] = {
    "++\n"
    "++",

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

    "+--------+-+----------------+-+----------------+-+--------+\n"
    "|        | |                +-+                | |        |\n"
    "|        ++++                                  | |        |\n"
    "|        ++++                             +----+ |        |\n"
    "|        ++++                             |+-----+    ++  |\n"
    "|        ++++              +----+         ||          ||  |\n"
    "+-----------+      ++      |+--+|  ++--+  ||  +-------+| ++\n"
    "| +--------+|      ||      ||++||  ||  |  ||  |     +--+ ||\n"
    "+-+   +---+||      ++      ||++||  ++--+  ||  |     +---+++\n"
    "|     |+-+|||              |+--+|         ||  +--------+| |\n"
    "|     || ++||              +----+         |+-----------+| |\n"
    "|     |+---+|                             +----+ +------+ |\n"
    "|     +-----+                                  | |        |\n"
    "|        +-+                +-+                | |        |\n"
    "|        +-+                +-+                | |        |\n"
    "|  +-----+ |    ++    +-----+ |    ++          | |        |\n"
    "|  +-++----+    ++    +-++----+    ++    +-----+ |        |\n"
    "|    ++                 ++               |+-+    |        |\n"
    "|    ||                 ||               || |  +-+        |\n"
    "++   |+-------------+   |+---------------+| +--+    +-----+\n"
    "||   |              |   |                 |      +--+     |\n"
    "++   +---+ +--------+   +---+ +-----------+  +---+   +----+\n"
    "|        | |                | |              |       |    |\n"
    "|        | |                | |              +-+ +---+    |\n"
    "|        | |                | |                | |        |\n"
    "|        | |                | |                | |        |\n"
    "|        | |                | |                | |        |\n"
    "+--------+-+----------------+-+----------------+-+--------+",
  };

  for (auto &shape : tst_shapes) {
    std::vector<std::string> pieces = break_piece(shape);
    for (auto &p : pieces) {
      std::cout << p << "\n***\n";
    }

    std::cout << "*********************" << std::endl;
  }
}

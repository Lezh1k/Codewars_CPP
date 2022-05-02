//#include <algorithm>
//#include <array>
//#include <iostream>
//#include <numeric>
//#include <map>
//#include <queue>
//#include <set>
//#include <vector>

//using namespace std;
//vector<string> break_piece(const string &shape);
//vector<string> split_lines(const string &to_split, const string &delimiter);
//// seems like we don't need this
//class GameCoord {
//private:
//  int m_row;
//  int m_col;

//public:
//  GameCoord() : m_row(0), m_col(0) {}
//  GameCoord(int row, int col) : m_row(row), m_col(col) {}
//  ~GameCoord() = default;

//  int row() const { return m_row; }
//  int col() const { return m_col; }

//  friend inline bool operator==(const GameCoord &l, const GameCoord &r) {
//    return l.m_row == r.m_row && l.m_col == r.m_col;
//  }
//};

//inline bool operator<(const GameCoord &lhs, const GameCoord &rhs) {
//  if (lhs.row() != rhs.row())
//    return lhs.row() < rhs.row();
//  return lhs.col() < rhs.col();
//}
//inline bool operator>(const GameCoord &lhs, const GameCoord &rhs) {
//  return rhs < lhs;
//}
//inline bool operator<=(const GameCoord &lhs, const GameCoord &rhs) {
//  return !(lhs > rhs);
//}
//inline bool operator>=(const GameCoord &lhs, const GameCoord &rhs) {
//  return !(lhs < rhs);
//}

////////////////////////////////////////////////////////////////

//class GameField {
//private:
//  vector<string> m_field;

//  /// \brief get_neighbors - returns current coordinate's possible neighbors on
//  /// game field \param cc - current coordinates \return vector of coordinates
//  std::vector<GameCoord> get_neighbors(const GameCoord &cc);

//public:
//  GameField() = delete;
//  explicit GameField(const string &shape);

//  char operator()(size_t row, size_t col) const;
//  friend std::ostream &operator<<(std::ostream &os, const GameField &gf);
//};
////////////////////////////////////////////////////////////////

//std::ostream &operator<<(std::ostream &os, const GameField &gf) {
//  for (auto line : gf.m_field) {
//    os << line << std::endl;
//  }
//  return os;
//}
////////////////////////////////////////////////////////////////

//std::vector<GameCoord> GameField::get_neighbors(const GameCoord &cc) {
//  static std::vector<int> cdeltas = {-1, 1};
//  std::vector<int> dr(cdeltas), dc(cdeltas);
//  char c = (*this)(cc.row(), cc.col());
//  std::vector<GameCoord> rlst;

//  switch (c) {
//  case '-':
//    dr.clear();
//    break;
//  case '|':
//    dc.clear();
//    break;
//  case '+':
//    break; // do nothing
//  default:
//    throw std::invalid_argument("invalid value in current field position");
//  }

//  for (auto d : dr) {
//    if (cc.row() + d < 0 || static_cast<size_t>(cc.row() + d) >= this->m_field.size()) {
//      continue;
//    }
//    rlst.push_back(GameCoord(cc.row() + d, cc.col()));
//  }

//  for (auto d : dc) {
//    if (cc.col() + d < 0 || static_cast<size_t>(cc.col() + d) >= m_field[cc.row()].size()) {
//      continue;
//    }
//    rlst.push_back(GameCoord(cc.row(), cc.col() + d));
//  }

//  // remove all empty cells from result
//  rlst.erase(std::remove_if(rlst.begin(), rlst.end(), [this](const GameCoord &c) {
//    return std::isspace((*this)(c.row(),c.col()));
//  }), rlst.end());

//  return rlst;
//}
////////////////////////////////////////////////////////////////

//GameField::GameField(const string &shape) :
//  m_field(split_lines(shape, "\n")) {

//  //todo move this somewhere. cause here in constructor we will init vertex
//  std::queue<GameCoord> q;
//  std::set<GameCoord> used;
//  std::map<GameCoord, GameCoord> parents;

//  GameCoord target = GameCoord(0, 0);

//  used.insert(target);
//  parents[target] = GameCoord(-1,-1);
//  q.push(target);

//  while (!q.empty()) {
//    GameCoord cc = q.front();
//    q.pop();

//    std::vector<GameCoord> neighbors = get_neighbors(cc);
//    for (auto nei : neighbors) {
//      if (used.find(nei) != used.end())
//        continue;

//      used.insert(nei);
//      parents[nei] = cc;
//      q.push(nei);
//    }
//  }

//  std::cout << "HOHOHOH" << std::endl;
//}
////////////////////////////////////////////////////////////////

//char GameField::operator()(size_t row, size_t col) const {
//  return m_field[row][col];
//}

//vector<string> break_piece(const string &shape) {
//  (void)shape;
//  GameField gf(shape);
//  std::cout << gf << std::endl;
//  return vector<string>();
//}
////////////////////////////////////////////////////////////////

//vector<string> split_lines(const string &to_split, const string &delimiter) {
//  vector<string> res;
//  size_t first, second = 0;
//  while ((first = to_split.find_first_not_of(delimiter, second)) !=
//         std::string::npos) {
//    second = to_split.find_first_of(delimiter, first);
//    if (first != second) {
//      res.push_back(to_split.substr(first, second - first));
//    }
//  }
//  return res;
//}
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

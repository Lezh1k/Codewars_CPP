#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <set>
#include <vector>
/*
*** Lift Rules ***
The Lift only goes up or down!
Each floor has both UP and DOWN Lift-call buttons (except top and ground floors
which have only DOWN and UP respectively)
The Lift never changes direction until there are no more people wanting to get
on/off in the direction it is already travelling.
When empty the Lift tries to be smart.
For example, If it was going up then it may continue up to collect the
highest floor person wanting to go down.
If it was going down then it may continue down to collect the lowest
floor person wanting to go up.
The Lift has a maximum capacity of people.
When called, the Lift will stop at a floor even if it
is full, although unless somebody gets off nobody else can get on!
If the lift is empty, and no people are waiting, then it will return to the
ground floor
*/

enum lift_dir_t { LD_UP = 1, LD_DOWN = -1 };
//////////////////////////////////////////////////////////////

class Game
{
 private:
  std::multiset<int> m_lift_persons;
  int m_lift_direction;
  int m_lift_current_floor;

  std::vector<std::vector<int>> m_floors_passengers;
  std::vector<int> m_visited_floors;
  size_t m_lift_capacity;

  bool get_floor_with_passangers_following_lift_direction(
      int current_dir,
      int& dst_floor,
      bool same_direction) const
  {
    // if (m_lift_direction == LD_UP) {
    int start = m_lift_current_floor;
    int end = static_cast<int>(m_floors_passengers.size());
    std::function<bool(int, int)> comp = std::less<int>();
    int df = 1;
    // }
    if (current_dir == LD_DOWN) {
      start = m_lift_current_floor;
      end = -1;
      comp = std::greater<int>();
      df = -1;
    }

    if (!same_direction) {
      end -= df;
      std::swap(start, end);
      df = -df;
    }

    for (int fi = start; fi != end; fi += df) {
      for (int person_tfi : m_floors_passengers[fi]) {
        if (same_direction && comp(person_tfi, fi))
          continue;
        if (!same_direction && !comp(person_tfi, fi))
          continue;

        dst_floor = fi;
        return true;
      }
    }
    return false;
  }
  //////////////////////////////////////////////////////////////

  int get_target_floor() const
  {
    int cd = m_lift_direction;
    for (int i = 0; i < 2; ++i) {
      if (!m_lift_persons.empty()) {
        int dst_highest = *m_lift_persons.rbegin();
        int dst_lowest = *m_lift_persons.begin();
        int dst_floor = cd == LD_UP ? dst_highest : dst_lowest;
        if (m_lift_current_floor != dst_floor) {
          return dst_floor;
        }
      }  // if (!m_lift_persons.empty())

      int fi = static_cast<int>(m_floors_passengers.size());
      bool has_passangers_same_dir =
          get_floor_with_passangers_following_lift_direction(cd, fi, true);
      if (has_passangers_same_dir) {
        return fi;
      }

      fi = 0;
      bool has_passengers_opposite_dir =
          get_floor_with_passangers_following_lift_direction(cd, fi, false);
      if (has_passengers_opposite_dir) {
        return fi;
      }
      // change lift direction and try all checks again!
      cd = -cd;
    }

    // if we are here - lift is not needed anymore and should go to the ground
    // floor
    return -1;
  }
  //////////////////////////////////////////////////////////////

 public:
  Game(const std::vector<std::vector<int>>& queues, int capacity)
      : m_lift_direction(LD_UP),
        m_lift_current_floor(0),
        m_floors_passengers(queues),
        m_lift_capacity(capacity)
  {
    m_visited_floors.push_back(0);
  }
  //////////////////////////////////////////////////////////////

  bool next_move()
  {
    std::function<bool(int, int)> comp = std::less_equal<int>();
    if (m_lift_direction == LD_DOWN) {
      comp = std::greater_equal<int>();
    }

    // 1. check that anybody in lift wants to get out
    // if yes - let them go
    bool stop = false;
    auto p_out = m_lift_persons.equal_range(m_lift_current_floor);
    if (p_out.first != m_lift_persons.end() &&
        *p_out.first == m_lift_current_floor) {
      stop = true;
      m_lift_persons.erase(p_out.first, p_out.second);
    }

    // 2. check that anybody on a floor wants to get in
    // if yes - let enter as many people as possible
    for (int& person_tfi : m_floors_passengers[m_lift_current_floor]) {
      if (comp(person_tfi, m_lift_current_floor))
        continue;
      // found person that needs this lift on current floor
      stop = true;
      if (m_lift_persons.size() >= m_lift_capacity)
        break;  // no need to check next persons. lift is full

      m_lift_persons.insert(person_tfi);
      person_tfi = -1;
    }

    // remove all entered to the lift passangers from queue
    m_floors_passengers[m_lift_current_floor].erase(
        std::remove_if(m_floors_passengers[m_lift_current_floor].begin(),
                       m_floors_passengers[m_lift_current_floor].end(),
                       [](int x) { return x == -1; }),
        m_floors_passengers[m_lift_current_floor].end());

    if (stop && m_visited_floors.back() != m_lift_current_floor) {
      m_visited_floors.push_back(m_lift_current_floor);
    }

    int tf = get_target_floor();
    if (tf == -1) {
      if (m_lift_current_floor != 0) {
        m_visited_floors.push_back(0);
      }
      return false;
    }

    if (tf == m_lift_current_floor) {
      m_lift_direction *= -1;
      return next_move();
    }

    m_lift_direction = tf > m_lift_current_floor ? LD_UP : LD_DOWN;

    m_lift_current_floor += m_lift_direction;
    return m_lift_current_floor >= 0;
  }
  //////////////////////////////////////////////////////////////

  const std::vector<int>& visited_floors() const
  {
    return m_visited_floors;
  }
  //////////////////////////////////////////////////////////////

  friend std::ostream& operator<<(std::ostream& out, const Game& g);
};
//////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& out, const Game& g)
{
  int tf = g.get_target_floor();
  std::cout << "cf : " << g.m_lift_current_floor << std::endl;
  std::cout << "tf : " << tf << std::endl;
  for (int cf = g.m_floors_passengers.size() - 1; cf >= 0; --cf) {
    out << cf << ".";
    if (g.m_lift_current_floor == cf) {
      out << "*\t{";
      for (auto p : g.m_lift_persons) {
        out << p << ", ";
      }
      out << "}\t";
    } else {
      out << "\t\t";
    }
    out << "[";
    for (auto p : g.m_floors_passengers[cf]) {
      out << p << ", ";
    }
    out << "]" << std::endl;
  }
  return out;
}
//////////////////////////////////////////////////////////////

/*
*** People Rules ***
People are in "queues" that
represent their order of arrival to wait for the Lift.
All people can press the UP/DOWN Lift-call buttons.
Only people going the same direction as the Lift may enter it.
Entry is according to the "queue" order, but those unable to enter do
not block those behind them that can.
If a person is unable to enter a full Lift,
they will press the UP/DOWN Lift-call button again after it has departed without
them

*** Kata Task ***
Get all the people to the floors they want to go to while obeying
the Lift rules and the People rules Return a list of all floors that the Lift
stopped at (in the order visited!) NOTE: The Lift always starts on the ground
floor (and people waiting on the ground floor may enter immediately)
 * */

std::vector<int> the_lift(const std::vector<std::vector<int>>& queues,
                          int capacity)
{
  int i = 1;
  Game g(queues, capacity);
  while (g.next_move() && i != 0) {
    std::cout << g << std::endl;
    std::cin >> i;
  }
  return g.visited_floors();
}
//////////////////////////////////////////////////////////////


void lift_main() {
  std::vector<std::vector<int>> queues;
  queues = {
      {
       3, 3,
       3, 3,
       3, 3,
       },
      {},
      {},
      {},
      {},
      {},
      {}
  };
  auto res = the_lift(queues, 5);
  for (auto f : res) {
    std::cout << f << ", ";
  }
  std::cout << std::endl;
}

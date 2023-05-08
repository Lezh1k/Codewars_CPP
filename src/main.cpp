#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
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

class Game {
private:
  const int m_lift_capacity;
  lift_dir_t m_lift_direction;
  int m_lift_current_floor;

  // kind of hack because we have const std::vector<std::vector<int>> in arg
  // so we can not modify people queues.
  std::vector<int> m_queue_tops;

public:
  Game(const std::vector<std::vector<int>> &queues, int capacity)
      : m_lift_capacity(capacity), m_lift_current_floor(0),
        m_queue_tops(queues.size()) {}

  bool next_move() { return false; }
};
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

std::vector<int> the_lift(const std::vector<std::vector<int>> &queues,
                          int capacity) {
  Game g(queues, capacity);
  return {};
}
//////////////////////////////////////////////////////////////

#ifdef _UNIT_TESTS_
int main_tests(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif
//////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
#ifdef _UNIT_TESTS_
  return main_tests(argc, argv);
#else
  for (int i = 0; i < argc; ++i) {
    std::cout << i << ": " << argv[i] << "\n";
  }
  std::vector<std::vector<int>> queues;
  queues = {{}, {}, {5, 5, 5}, {}, {}, {}, {}};
  the_lift(queues, 5);
#endif
}
//////////////////////////////////////////////////////////////

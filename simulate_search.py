from learn import MarkovAgent
from search import *

simulator = SearchSimulation()
observations = simulator.observations(100000, 20)
mark = MarkovAgent(observations)
mark.learn()

class AISearch(Search):
  def update_location(self):
    self.location = mark.policy[self.state()]

def print_ai_search():
  sorted_array = simulator._random_sorted_array(20)
  target_int = random.choice(sorted_array)
  search = AISearch(sorted_array, target_int)
  observation = simulator.observation(len(sorted_array), supplied_search = search)
  for location in search.path:
    print "Found {0} at index {1}. Looking for {2}".format(
      search.array[location],
      location,
      search.target
    )

print_ai_search()
# Found 814 at index 15. Looking for 30
# Found 25 at index 1. Looking for 30
# Found 650 at index 12. Looking for 30
# Found 97 at index 4. Looking for 30
# Found 30 at index 2. Looking for 30
# >>> print_ai_search()
# Found 532 at index 11. Looking for 999
# Found 795 at index 18. Looking for 999
# Found 999 at index 19. Looking for 999
# >>> print_ai_search()
# Found 129 at index 4. Looking for 5
# Found 28 at index 1. Looking for 5
# Found 5 at index 0. Looking for 5
# >>> print_ai_search()
# Found 491 at index 9. Looking for 351
# Found 64 at index 1. Looking for 351
# Found 326 at index 6. Looking for 351
# Found 407 at index 8. Looking for 351
# Found 351 at index 7. Looking for 351
# >>> print_ai_search()
# Found 870 at index 14. Looking for 899
# Found 972 at index 19. Looking for 899
# Found 905 at index 16. Looking for 899
# Found 899 at index 15. Looking for 899
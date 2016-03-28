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

print_ai_search() #more iterations seems to have room to improve performance
# >>> print_ai_search()
# Found 344 at index 9. Looking for 733
# Found 851 at index 18. Looking for 733
# Found 676 at index 13. Looking for 733
# Found 772 at index 17. Looking for 733
# Found 749 at index 15. Looking for 733
# Found 733 at index 14. Looking for 733
# >>> print_ai_search()
# Found 490 at index 11. Looking for 908
# Found 866 at index 18. Looking for 908
# Found 908 at index 19. Looking for 908
# >>> print_ai_search()
# Found 414 at index 9. Looking for 665
# Found 928 at index 18. Looking for 665
# Found 580 at index 13. Looking for 665
# Found 912 at index 17. Looking for 665
# Found 700 at index 15. Looking for 665
# Found 665 at index 14. Looking for 665
# >>> print_ai_search()
# Found 358 at index 8. Looking for 328
# Found 45 at index 1. Looking for 328
# Found 193 at index 5. Looking for 328
# Found 328 at index 7. Looking for 328
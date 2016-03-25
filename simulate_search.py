from learn import MarkovAgent
from search import *

simulator = SearchSimulation()
mark = MarkovAgent(simulator.observations(100000, 25))
mark.learn()

class AISearch(Search):
  def update_location(self):
    self.location = mark.policy[self.state]

def print_ai_search():
  sorted_array = simulator._random_sorted_array(25)
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
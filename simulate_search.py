from learn import MarkovAgent
from search import *

simulator = SearchSimulation()
mark = MarkovAgent(simulator.observations(100000, 25))
mark.learn()

class AISearch(Search):
  def update_location(self):
    self.location = mark.policy[self.state]

sorted_array = simulator._random_sorted_array(25)
target_int = random.choice(sorted_array)

ai = AISearch(sorted_array, target_int)
simulator.simulate_search(25, ai)
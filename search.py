from learn import MarkovAgent
import random_search
import binary_search
import random

# simulate agent search
def perform_agent_search(agent):
  # create random values and choose the target at random
  sorted_array = binary_search.random_sorted_ints(25)
  target_int = random.choice(sorted_array)

  # initialize state and location
  location = random.choice(range(25))
  state = binary_search.parse_state(sorted_array, location, target_int)

  # loop through the search
  while (state != 50):
    print "Found {0} at index {1}. Looking for {2}".format(
      sorted_array[location],
      location,
      target_int
    )
    location = int(agent.policy[state])
    state = binary_search.parse_state(sorted_array, location, target_int)
  print "Found {0} at index {1}!".format(target_int, location)
  print (sorted_array)

# note that dimensions can be inferred from observations if desired
# but it is frequently the case that these are known for the application
DIMENSIONS = { 'state_count': 51, 'action_count': 25 }
observations = binary_search.observations(250000) + random_search.observations(100000)
mark = MarkovAgent(observations, DIMENSIONS)
mark.learn()

# view search performance of our agent
perform_agent_search(mark)
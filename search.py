# new encoding breaks this and search_data too
# state parsed in these files is 'string' state not encoded state

from learn import MarkovAgent
import search_data
import random

# simulate agent search
def simulate_agent_search(agent, array_length):
  # create random values and choose the target at random
  sorted_array = search_data.random_sorted_ints(array_length)
  target_int = random.choice(sorted_array)

  # initialize state and location
  location = random.choice(range(array_length))
  state = search_data.parse_state(sorted_array, location, target_int)

  # loop through the search
  while (state != (array_length * 2)):
    print "Found {0} at index {1}. Looking for {2}".format(
      sorted_array[location],
      location,
      target_int
    )

    parsed_state = agent.state_action_encoder.state_to_int[state]
    location = int(agent.policy[parsed_state])
    state = search_data.parse_state(sorted_array, location, target_int)
  print "Found {0} at index {1}!".format(target_int, location)


# create an agent trained on data
ARRAY_LENGTH = 25
DIMENSIONS = { 'state_count': (ARRAY_LENGTH * 2 + 1), 'action_count': ARRAY_LENGTH }
observations = search_data.observations(10, ARRAY_LENGTH)

mark = MarkovAgent(observations)
mark.learn()

# view search performance of our agent
simulate_agent_search(mark, ARRAY_LENGTH)

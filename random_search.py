import random

def observations(n):
  print "SIMULATING {0} RANDOM SEARCHES".format(n)
  observations = []
  for i in range(n): observations.append(observation())
  observations.append(trap_state())
  return observations

def observation():
  # initialize array of transitions
  transitions = []

  # create random values and choose the target at random
  sorted_array = random_sorted_ints(25)
  target_int = random.choice(sorted_array)

  # initialize the search
  initial_location = random.choice(range(25))
  state = parse_state(sorted_array, initial_location, target_int)

  # randomly choose locations until we find the target int
  while (state != 50):
    location = random.choice(range(25))
    state_ = parse_state(sorted_array, location, target_int)

    transitions.append({
      'state': state,
      'action': location,
      'state_': state_
    })

    state = state_

  #return observation
  if (len(transitions) == 0):
    transitions.append({ 'state': 50, 'action': 0, 'state_': 50 })
  return {
    'state_transitions': transitions,
    'reward': -len(transitions)
  }

def parse_state(random_ints, current_location, target_int):
  if (random_ints[current_location] < target_int):
    return current_location
  if (random_ints[current_location] > target_int):
    return (current_location + 25)
  if (random_ints[current_location] == target_int):
    return 50

def random_sorted_ints(length):
  random_ints = random.sample(range(0, 1000), length)
  random_ints.sort()
  return random_ints

def trap_state():
  transitions = map(
    lambda i: { 'state': 50, 'action': i, 'state_': 50 },
    range(25)
  )
  return {
    'state_transitions': transitions,
    'reward': 100
  }
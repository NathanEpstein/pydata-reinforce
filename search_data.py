import random

def observations(n, array_length):
  print "SIMULATING {0} SEARCHES".format(n)
  observations = []
  for i in range(n): observations.append(observation(array_length))
  observations.append(trap_state(array_length))
  return observations

def observation(array_length):
  # initialize array of transitions
  transitions = []

  # create random values and choose the target at random
  sorted_array = random_sorted_ints(array_length)
  target_int = random.choice(sorted_array)

  # initialize the search
  search_type = random.choice(['binary', 'linear', 'random'])
  if search_type == 'binary':
    search = BinarySearch(sorted_array, target_int)
  if search_type == 'random':
    search = RandomSearch(sorted_array, target_int)
  if search_type == 'linear':
    search = LinearSearch(sorted_array, target_int)

  state = parse_state(sorted_array, search.location, target_int)

  # randomly choose locations until we find the target int
  while (state != (array_length * 2)):
    location = search.update()
    state_ = parse_state(sorted_array, location, target_int)

    transitions.append({
      'state': state,
      'action': location,
      'state_': state_
    })

    state = state_

  #return observation
  if (len(transitions) == 0): return observation(array_length)
  return {
    'state_transitions': transitions,
    'reward': -len(transitions)
  }

def parse_state(array, current_location, target_int):
  if (array[current_location] < target_int):
    return current_location
  if (array[current_location] > target_int):
    return (current_location + len(array))
  if (array[current_location] == target_int):
    return len(array * 2)

def random_sorted_ints(array_length):
  random_ints = random.sample(range(0, 1000), array_length)
  random_ints.sort()
  return random_ints

def trap_state(array_length):
  transitions = map(
    lambda i: { 'state': (array_length * 2), 'action': i, 'state_': (array_length * 2) },
    range(array_length)
  )
  return {
    'state_transitions': transitions,
    'reward': array_length
  }

class BinarySearch:
  def __init__(self, array, target):
    self.a = 0
    self.b = len(array)
    self.location = random.choice(range(len(array)))
    self.array = array
    self.target = target

  def update(self):
    if self.array[self.location] < self.target:
      self.a = self.location
    else:
      self.b = self.location
    self.location = (self.a + self.b) / 2
    return self.location

class LinearSearch:
  def __init__(self, array, target):
    self.location = random.choice(range(len(array)))
    self.array = array
    self.target = target

  def update(self):
    if self.array[self.location] < self.target:
      self.location += 1
    else:
      self.location -= 1
    return self.location

class RandomSearch:
  def __init__(self, array, target):
    self.array_length = len(array)
    self.location = random.choice(range(self.array_length))

  def update(self):
    self.location = random.choice(range(self.array_length))
    return self.location



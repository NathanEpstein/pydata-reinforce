import random

# parent search class, methods for updating relevant features as search progresses
class Search:
  def __init__(self, array, target):
    self.path = []
    self.array = array
    self.target = target
    self.initialize_features()

  def initialize_features(self):
    self.location = random.choice(range(len(self.array)))
    self._update_direction()
    self.floor = 0
    self.ceil = len(self.array)

  def state(self):
    if (self.array[self.location] == self.target):
      return 'TARGET_FOUND'
    features = [
      str(self.location),
      self.direction,
      str(self.floor),
      str(self.ceil)
    ]
    return ':'.join(features)

  def update(self):
    self.update_location() #supplied by child class
    self.path.append(self.location)
    self._update_direction_and_bounds()
    return self.location

  def _update_direction_and_bounds(self):
    self._update_direction()
    self._update_bounds()

  def _update_direction(self):
    if self.array[self.location] < self.target:
      self.direction = 'up'
    else:
      self.direction = 'down'

  def _update_bounds(self):
    if self.direction == 'up':
      self.floor = self.location
    else:
      self.ceil = self.location


# specific search classes inherit from Search and supply update_location method
class BinarySearch(Search):
  def update_location(self):
    self.location = (self.ceil + self.floor) / 2

class LinearSearch(Search):
  def update_location(self):
    if self.direction == 'up':
      self.location += 1
    else:
      self.location -= 1

class RandomSearch(Search):
  def update_location(self):
    self.location = random.choice(range(len(self.array)))

# simulation class which uses searches to create training data
class SearchSimulation:
  def observation(self, array_length, supplied_search = None):
    if supplied_search is None:
      search = self._search_of_random_type(array_length)
    else:
      search = supplied_search

    transitions = []

    while (search.state() != 'TARGET_FOUND'):
      state = search.state()
      action = search.update()
      state_ = search.state()

      transitions.append({
        'state': state,
        'action': action,
        'state_': state_
      })

    if len(transitions) == 0:
      return self.observation(array_length, supplied_search = search)
    else:
      return {
        'reward': -len(transitions),
        'state_transitions': transitions
      }

  def observations(self, n, array_length):
    observations = []
    for i in range(n):
      observations.append(self.observation(array_length))
    observations.append(self._trap_state(array_length))
    return observations

  def _trap_state(self, array_length):
    state_transitions = []
    for i in range(array_length):
      state_transitions.append({
        'state': 'TARGET_FOUND',
        'action': i,
        'state_': 'TARGET_FOUND'
      })
    return {
      'reward': 1,
      'state_transitions': state_transitions
    }

  def _search_of_random_type(self, array_length):
    sorted_array = self._random_sorted_array(array_length)
    target_int = random.choice(sorted_array)

    search_type = random.choice(['binary', 'linear', 'random'])
    if search_type == 'binary':
      search = BinarySearch(sorted_array, target_int)
    if search_type == 'random':
      search = RandomSearch(sorted_array, target_int)
    if search_type == 'linear':
      search = LinearSearch(sorted_array, target_int)

    return search

  def _random_sorted_array(self, length):
    random_ints = random.sample(range(0, 1000), length)
    random_ints.sort()
    return random_ints

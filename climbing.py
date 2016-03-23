from learn import MarkovAgent

# note that dimensions can be inferred from observations if desired
# but it is frequently the case that these are known for the application
DIMENSIONS = {
  'state_count': 5,
  'action_count': 2
}

observations = [
  { 'state_transitions': [
      { 'state': 1, 'action': 1, 'state_': 2 },
      { 'state': 2, 'action': 0, 'state_': 1 },
      { 'state': 1, 'action': 0, 'state_': 0 }
    ],
    'reward': 0
  },
  { 'state_transitions': [
      { 'state': 1, 'action': 1, 'state_': 2 },
      { 'state': 2, 'action': 1, 'state_': 3 },
      { 'state': 3, 'action': 1, 'state_': 4 },
    ],
    'reward': 1
  }
]

trap_states = [
  {
    'state_transitions': [
      { 'state': 0, 'action': 0, 'state_': 0 },
      { 'state': 0, 'action': 1, 'state_': 0 }
    ],
    'reward': 0
  },
  {
    'state_transitions': [
      { 'state': 4, 'action': 0, 'state_': 4 },
      { 'state': 4, 'action': 1, 'state_': 4 },
    ],
    'reward': 1
  },
]

observations += trap_states

mark = MarkovAgent(observations, DIMENSIONS)
mark.learn()

# mark correctly learns that the optimal strategy is to always go up
print(mark.policy)

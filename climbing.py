from learn import MarkovAgent

observations = [
  { 'state_transitions': [
      { 'state': 'low', 'action': 'climb', 'state_': 'mid' },
      { 'state': 'mid', 'action': 'climb', 'state_': 'high' },
      { 'state': 'high', 'action': 'sink', 'state_': 'mid' },
      { 'state': 'mid', 'action': 'sink', 'state_': 'low' },
      { 'state': 'low', 'action': 'sink', 'state_': 'bottom' }
    ],
    'reward': 0
  },
  { 'state_transitions': [
      { 'state': 'low', 'action': 'climb', 'state_': 'mid' },
      { 'state': 'mid', 'action': 'climb', 'state_': 'high' },
      { 'state': 'high', 'action': 'climb', 'state_': 'top' },
    ],
    'reward': 0
  }
]

trap_states = [
  {
    'state_transitions': [
      { 'state': 'bottom', 'action': 'sink', 'state_': 'bottom' },
      { 'state': 'bottom', 'action': 'climb', 'state_': 'bottom' }
    ],
    'reward': 0
  },
  {
    'state_transitions': [
      { 'state': 'top', 'action': 'sink', 'state_': 'top' },
      { 'state': 'top', 'action': 'climb', 'state_': 'top' },
    ],
    'reward': 1
  },
]

observations += trap_states

mark = MarkovAgent(observations)
mark.learn()

# mark correctly learns that the optimal strategy is to always go up
print(mark.policy)

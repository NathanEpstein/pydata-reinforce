from rewards import RewardParser
from transitions import TransitionParser
from policy import PolicyParser

class MarkovAgent:
  def __init__(self, observations, dimensions):
    self.reward_parser = RewardParser(observations, dimensions)
    self.transition_parser = TransitionParser(observations, dimensions)
    self.policy_parser = PolicyParser(dimensions)

  def learn(self):
    R = self.reward_parser.rewards()
    P = self.transition_parser.transition_probabilities()
    self.policy = self.policy_parser.policy(P, R)


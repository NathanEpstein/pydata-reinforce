from encoding import StateActionEncoder
from rewards import RewardParser
from transitions import TransitionParser
from policy import PolicyParser

class MarkovAgent:
  def __init__(self, observations):
    self.state_action_encoder = StateActionEncoder(observations)
    dimensions = self.state_action_encoder.parse_dimensions()

    # transform observation to int encoding before this
    self.reward_parser = RewardParser(observations, dimensions)
    self.transition_parser = TransitionParser(observations, dimensions)
    self.policy_parser = PolicyParser(dimensions)

  def learn(self):
    R = self.reward_parser.rewards()
    P = self.transition_parser.transition_probabilities()
    self.policy = self.policy_parser.policy(P, R)
    # transform policy from int encoding to string
    # i.e. have a separate method to access the string dict version of the policy...


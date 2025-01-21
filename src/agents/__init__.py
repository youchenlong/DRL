from .dqn import DQN
from .doubledqn import DoubleDQN
from .duelingdqn import DuelingDQN
from .reinforce import REINFORCE
from .reinforce_baseline import REINFORCEBaseline
from .actor_critic import ActorCritic
from .ddpg import DDPG
from .reinforce_discrete import REINFORCEDiscrete
from .reinforce_baseline_discrete import REINFORCEBaselineDiscrete
from .actor_critic_discrete import ActorCriticDiscrete

a_REGISTRY = {}
a_REGISTRY["dqn"] = DQN
a_REGISTRY["double_dqn"] = DoubleDQN
a_REGISTRY["dueling_dqn"] = DuelingDQN
a_REGISTRY["reinforce"] = REINFORCE
a_REGISTRY["reinforce_baseline"] = REINFORCEBaseline
a_REGISTRY["actor_critic"] = ActorCritic
a_REGISTRY["ddpg"] = DDPG
a_REGISTRY["reinforce_discrete"] = REINFORCEDiscrete
a_REGISTRY["reinforce_baseline_discrete"] = REINFORCEBaselineDiscrete
a_REGISTRY["actor_critic_discrete"] = ActorCriticDiscrete
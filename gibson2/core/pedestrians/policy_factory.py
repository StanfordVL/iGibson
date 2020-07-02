from core.pedestrians.linear import Linear
from core.pedestrians.orca import ORCA


def none_policy():
    return None


policy_factory = dict()
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy

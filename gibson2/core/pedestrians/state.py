class FullState(object):
    def __init__(self, px, py, theta, vx, vy, vr, radius, personal_space, gx, gy, gr, v_pref):
        self.px = px
        self.py = py
        self.theta = theta
        self.vx = vx
        self.vy = vy
        self.vr = vr
        self.radius = radius
        self.personal_space = personal_space
        self.gx = gx
        self.gy = gy
        self.gr = gr
        self.v_pref = v_pref

        self.position = (self.px, self.py, self.theta)
        self.goal_position = (self.gx, self.gy, self.gr)
        self.velocity = (self.vx, self.vy, self.vr)

    def __add__(self, other):
        return other + (self.px, self.py, self.theta, self.vx, self.vy, self.vr, self.radius, self.personal_space, self.gx, self.gy, self.gr, self.v_pref)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.theta, self.vx, self.vy, self.vr, self.radius, self.personal_space, self.gx, self.gy, self.gr,
                                          self.v_pref]])


class ObservableState(object):
    def __init__(self, px, py, theta, vx, vy, vr, radius, personal_space):
        self.px = px
        self.py = py
        self.theta = theta
        self.vx = vx
        self.vy = vy
        self.vr = vr
        self.radius = radius
        self.personal_space = personal_space

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)
        self.pose = (self.px, self.py, self.theta)

    def __add__(self, other):
        return other + (self.px, self.py, self.theta, self.vx, self.vy, self.vr, self.radius, self.personal_space)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.theta, self.vx, self.vy, self.vr, self.radius, self.personal_space]])


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
        #self.obstacles = obstacles

class TensorFlowState(object):
    def __init__(self, self_state, human_states):
        self.self_state = self_state
        self.human_states = human_states
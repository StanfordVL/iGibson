import numpy as np


class Filter(object):
    """
    A base class for filtering a noisy data stream in an online fashion.
    """

    def __init__(self):
        pass

    def estimate(self, observation):
        """
        Takes an observation and returns a de-noised estimate.
        :param observation: A current observation.
        :return: De-noised estimate.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets this filter. Default is no-op.
        """
        pass

    def dump_state(self):
        """
        :return Any: the state of the object other than what's not included in pybullet state.
        """
        return None

    def load_state(self, dump):
        """
        Load the state of the object other than what's not included in pybullet state.

        :param dump: Any: the dumped state
        """
        return


class MovingAverageFilter(Filter):
    """
    This class uses a moving average to de-noise a noisy data stream in an online fashion.
    This is a FIR filter.
    """

    def __init__(self, obs_dim, filter_width):
        """
        :param obs_dim: The dimension of the points to filter.
        :param filter_width: The number of past samples to take the moving average over.
        """
        self.filter_width = filter_width
        self.past_samples = []
        self.past_samples_sum = np.zeros(obs_dim)
        self.num_samples = 0

        super(MovingAverageFilter, self).__init__()

    def estimate(self, observation):
        """
        Do an online hold for state estimation given a recent observation.
        :param observation: New observation to hold internal estimate of state.
        :return: New estimate of state.
        """
        if self.num_samples == self.filter_width:
            val = self.past_samples.pop(0)
            self.past_samples_sum -= val
            self.num_samples -= 1
        self.past_samples.append(np.array(observation))
        self.past_samples_sum += observation
        self.num_samples += 1

        return self.past_samples_sum / self.num_samples

    def reset(self):
        # Clear internal state
        self.past_samples = []
        self.past_samples_sum *= 0.0
        self.num_samples = 0

    def dump_state(self):
        """
        :return Any: the state of the object other than what's not included in pybullet state.
        """
        return {
            "filter_width": self.filter_width,
            "past_samples": [item.tolist() for item in self.past_samples],
            "past_samples_sum": self.past_samples_sum.tolist(),
            "num_samples": self.num_samples,
        }

    def load_state(self, dump):
        """
        Load the state of the object other than what's not included in pybullet state.

        :param dump: Any: the dumped state
        """
        assert self.filter_width == dump["filter_width"], "filter width mismatch"
        self.past_samples = [np.array(item) for item in dump["past_samples"]]
        self.past_samples_sum = np.array(dump["past_samples_sum"])
        self.num_samples = dump["num_samples"]


class ExponentialAverageFilter(Filter):
    """
    This class uses an exponential average of the form y_n = alpha * x_n + (1 - alpha) * y_{n - 1}.
    This is an IIR filter.
    """

    def __init__(self, obs_dim, alpha=0.9):
        """
        :param obs_dim: The dimension of the points to filter.
        :param filter_width: The number of past samples to take the moving average over.
        """
        self.avg = np.zeros(obs_dim)
        self.num_samples = 0
        self.alpha = alpha

        super(ExponentialAverageFilter, self).__init__()

    def estimate(self, observation):
        """
        Do an online hold for state estimation given a recent observation.
        :param observation: New observation to hold internal estimate of state.
        :return: New estimate of state.
        """
        self.avg = self.alpha * observation + (1.0 - self.alpha) * self.avg
        self.num_samples += 1

        return np.array(self.avg)

    def dump_state(self):
        """
        :return Any: the state of the object other than what's not included in pybullet state.
        """
        return {
            "alpha": self.alpha,
            "avg": self.avg.tolist(),
            "num_samples": self.num_samples,
        }

    def load_state(self, dump):
        """
        Load the state of the object other than what's not included in pybullet state.

        :param dump: Any: the dumped state
        """
        assert self.alpha == dump["alpha"], "filter alpha mismatch"
        self.avg = np.array(dump["avg"])
        self.num_samples = dump["num_samples"]


class Subsampler(object):
    """
    A base class for subsampling a data stream in an online fashion.
    """

    def __init__(self):
        pass

    def subsample(self, observation):
        """
        Takes an observation and returns the observation, or None, which
        corresponds to deleting the observation.
        :param observation: A current observation.
        :return: The observation, or None.
        """
        raise NotImplementedError


class UniformSubsampler(Subsampler):
    """
    A class for subsampling a data stream uniformly in time in an online fashion.
    """

    def __init__(self, T):
        """
        :param T: Pick one every T observations.
        """
        self.T = T
        self.counter = 0

        super(UniformSubsampler, self).__init__()

    def subsample(self, observation):
        """
        Returns an observation once every T observations, None otherwise.
        :param observation: A current observation.
        :return: The observation, or None.
        """
        self.counter += 1
        if self.counter == self.T:
            self.counter = 0
            return observation
        return None


if __name__ == "__main__":
    f = MovingAverageFilter(3, 10)
    a = np.array([1, 1, 1])
    for i in range(500):
        print(f.estimate(a + np.random.normal(scale=0.1)))

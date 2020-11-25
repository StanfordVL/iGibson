import logging
import time


class Profiler(object):
    """
    A simple profiler for logging and debugging
    """

    def __init__(self, name, logger=None, level=logging.INFO, enable=True):
        self.name = name
        self.logger = logger
        self.level = level
        self.enable = enable

    def step(self, name):
        """ Returns the duration since last step/start """
        duration = self.summarize_step(
            start=self.step_start, step_name=name, level=self.level)
        now = time.time()
        self.step_start = now
        return duration

    def __enter__(self):
        self.start = time.time()
        self.step_start = time.time()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.enable:
            self.summarize_step(self.start)

    def summarize_step(self, start, level=None):
        """
        Summarize the step duration and fps
        """
        duration = time.time() - start
        if self.logger:
            level = level or self.level
            self.logger.log(self.level,
                            "{name}: {fps:.2f} fps, {duration:.5f} seconds".format(name=self.name,
                                                                                   fps=1 / duration,
                                                                                   duration=duration))
        else:
            print("{name}: {fps:.2f} fps, {duration:.5f} seconds".format(name=self.name,
                                                                         fps=1 / duration,
                                                                         duration=duration))
        return duration

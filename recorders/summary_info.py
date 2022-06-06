from abc import ABC, abstractmethod
import math

import logging

class Summary_Info(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.things = []
        self.info = {}
        self.summary_n_samples = {}

    def get_things(self):
        return self.things

    def get_info(self):
        return self.info

    def get_summary_n_samples(self):
        return self.summary_n_samples

    def get_thing_info(self, thing):
        return self.info(thing)

    def get_thing_summary_n_samples(self, thing):
        return self.summary_n_samples[thing]

    def set_things(self, things):
        self.things = things

    def set_info(self, info):
        self.things = list(info.keys())
        self.info = info

    def set_summary_n_samples(self, summary_n_samples):
        self.summary_n_samples = summary_n_samples

    def set_thing_info(self, thing, info):
        self.info[thing] = info

    def set_thing_summary_n_samples(self, thing, summary_n_samples):
        self.summary_n_samples[thing] = summary_n_samples




























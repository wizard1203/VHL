from abc import ABC, abstractmethod
import math

import logging
import pandas as pd

import torch

from utils.meter import AverageMeter

class Summary(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.things = []
        self.meter_dict = {}
        self.summary_n_samples_dict = {}

    def update(self, info, summary_n_samples=1):
        if type(summary_n_samples) == dict:
            for key, value in info.items():
                if key not in self.meter_dict:
                    self.things.append(key)
                    self.summary_n_samples_dict[key] = 0
                    self.meter_dict[key] = AverageMeter()

                if key not in summary_n_samples:
                    self.summary_n_samples_dict[key] += 1
                    self.meter_dict[key].update(value, 1)
                else:
                    self.summary_n_samples_dict[key] += summary_n_samples[key]
                    self.meter_dict[key].update(value, summary_n_samples[key])

        else:
            for key, value in info.items():
                if key not in self.meter_dict:
                    self.things.append(key)
                    self.summary_n_samples_dict[key] = 0
                    self.meter_dict[key] = AverageMeter()
                self.summary_n_samples_dict[key] += summary_n_samples
                self.meter_dict[key].update(value, summary_n_samples)

    def get_summary(self):
        summary_info = {}
        for key, meter in self.meter_dict.items():
            summary_info[key] = meter.avg
        return self.things, summary_info, self.summary_n_samples_dict

    def get_thing_summary(self, key):
        return self.meter_dict[key].avg, self.summary_n_samples_dict[key]

    def get_summary_n_samples_dict(self):
        return self.summary_n_samples_dict


    # def update_summary(self, dict):
    #     self.summary.update(dict)


























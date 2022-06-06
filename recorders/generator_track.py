import os
import logging

import torch

from configs.chooses import EPOCH, ITERATION

class generator_tracker(object):

    def __init__(self, args=None):
        self.things_to_track = ["generator_track"]

    def check_config(self, args, **kwargs):
        return True

    def generate_record(self, args, **kwargs):
        """ Here args means the overall args, not the *args """
        info_dict = {}

        if "Loss" in kwargs:
            info_dict["AuxData-Loss"] = kwargs["Loss"]
            info_dict["AuxData-Acc1"] = kwargs["Acc1"]
        else:
            pass

        if "align_domain_loss_value" in kwargs:
            info_dict["AlignDomain-Loss"] = kwargs["align_domain_loss_value"]

        if "align_cls_loss_value" in kwargs:
            info_dict["AlignClass-Loss"] = kwargs["align_cls_loss_value"]

        if "noise_cls_loss_value" in kwargs:
            info_dict["NoiseContrast-Loss"] = kwargs["noise_cls_loss_value"]

        if "PredShift_Acc1" in kwargs:
            info_dict["PredShift-Loss"] = kwargs["PredShift_Loss"]
            info_dict["PredShift-Acc1"] = kwargs["PredShift_Acc1"]
        else:
            pass

        logging.info('generator Losses TRACK::::   {}'.format(
            info_dict
        ))
        return info_dict

    def get_things_to_track(self):
        return self.things_to_track















from abc import ABC, abstractmethod
import math

import logging
import pandas as pd

import torch
from .build import create_trackers
from .summary import Summary
from .summary_info import Summary_Info
from utils.meter import AverageMeter, MaxMeter, MinMeter
from configs.chooses import MAX, MIN, AVG, NEW, OLD



def construct_query(info):
    query = ""
    for i, (key, value) in enumerate(info.items()):
        if i != 0:
            query += " & "
        if value is None or value == "None" :
            query += " {}.isnull() ".format(key) 
        else:
            query += " {} == {} ".format(key, value) 
    return query


class Recorder(object):

    def __init__(self, args):
        self.args = args
        self.summary = Summary()
        self.tracker_dict = create_trackers(args)
        self.reset()

    def reset(self):
        self.record = pd.DataFrame()
        self.reset_summary()

    def reset_summary(self, args=None, **kwargs):
        self.summary.reset()

    def update_record(self, key_info, thing, summary_n_samples=None, args=None, **kwargs):
        info_dict = self.generate_record(thing, args, **kwargs)
        if info_dict is None:
            return
        if summary_n_samples is None:
            summary_n_samples = 1
        else:
            pass
        self.summary.update(info_dict, summary_n_samples=summary_n_samples)

        if args.record_dataframe:
            for key in key_info:
                if key not in self.record.columns:
                    self.record.insert(loc=0, column=key, value=None)
            query_index = self.record.query(construct_query(key_info)).index
            if len(query_index) > 0:
                # logging.info("Query index bigger than 1, check it : {}, record: {}".format(
                #     key_info, self.record, 
                # ))
                self.record.loc[query_index, list(info_dict.keys())] = list(info_dict.values())
            else:
                info_dict.update(key_info)
                self.record = self.record.append(info_dict, ignore_index=True)
            # logging.info("In update_record.....info_dict is: {}, attain record as: {}".format(
            #     info_dict, self.record
            # ))

    def add_record(self, new_record, summary_n_samples=None):
        self.record = self.record.append(new_record, ignore_index=True)
        new_summary_dict = dict(new_record.mean())
        if summary_n_samples is None:
            self.summary.update(new_summary_dict, summary_n_samples=len(new_record))
            logging.warning(" WARNING: The n_samples should be indicated!!!")
        else:
            self.summary.update(new_summary_dict, summary_n_samples=summary_n_samples)

    def get_record(self, args=None, **kwargs):
        return self.record

    def generate_record(self, thing, args=None, **kwargs):
        """ Here args means the overall args, not the *args """
        if thing not in self.tracker_dict:
            logging.error("ERROR!! {} is not registered in the recorder!!".format(
                thing
            ))
            raise RuntimeError

        if not self.tracker_dict[thing].check_config(args, **kwargs):
            logging.info(f"ERROR!!!!!!! thing: {thing} check config is not True ")
            return None
        info_dict = self.tracker_dict[thing].generate_record(args, **kwargs)
        return info_dict

    def get_summary(self, args=None, **kwargs):
        return self.summary.get_summary()

    def get_thing_summary(self, key, args=None, **kwargs):
        return self.summary.get_thing_summary(key)

    def add_summary(self):
        logging.error("ERROR!!! Do not only update summary but not record....")
        raise NotImplementedError

    def get_n_samples_dict(self):
        return self.summary.get_n_samples_dict()



    def get_wandb_summary(self, args=None, **kwargs):
        pass

    def get_wandb_record(self, args=None, **kwargs):
        pass




import os
from copy import deepcopy

import logging
import pandas as pd
import wandb

from recorders.recorder import Recorder
from recorders.summary import Summary
from recorders.summary_info import Summary_Info

from .meter import AverageMeter
from .wandb_util import wandb_log


class RuntimeTracker(object):
    """Tracking the runtime stat for local training."""

    # def __init__(self, metrics_to_track=["top1"], on_cuda=True):
    def __init__(
        self,
        mode="Train",
        things_to_metric=["top1"],
        timer=None,
        on_cuda=True,
        id=None,
        args=None
    ):
        self.mode = mode
        self.things_to_metric = things_to_metric
        self.on_cuda = on_cuda
        self.metrics_n_samples = 0
        # self.time_stamp = 0
        self.id = id
        self.stat = None
        self.args = args
        self.timer = timer

        self.global_recorder = Recorder(args)
        # self.time_info_locals = dict((client_index, {"epoch": 0, "iteration": 0}) \
        #     for client_index in range(self.args.client_num_in_total))

        # will be reset after one communication at the end of each epoch, or round.
        # this local recorder can be used for both clients and servers.
        self.local_recorder = Recorder(args)
        self.summary_global = Summary()
        # self.sub_recorder_dict = create_recorder(self.args)
        # self.local_recorder_things_dict = {}
        self.reset()

        # This dict is used to record the history uploaded to wandb. 
        # will not be reset forever.
        self.wandb_summary_dict = {}


    def reset(self):
        logging.info(f"tracker reset.......")
        self.local_recorder.reset()
        self.summary_global.reset()

        # local: from different clients
        self.summary_info_locals = dict((client_index, Summary_Info()) for client_index in range(self.args.client_num_in_total))
        self.summary_info_server = Summary_Info()

        self.stat = dict((name, AverageMeter()) for name in self.things_to_metric)
        self.metrics_n_samples = 0


    def get_metrics_performance(self):
        return [self.stat[thing].avg for thing in self.things_to_metric]

    def update_metrics(self, metric_stat, metrics_n_samples):
        # if reset_metrics:
        #     for thing in self.things_to_metric:
        #         self.stat[thing].reset() 
        if metrics_n_samples == 0 or metrics_n_samples < 0:
            logging.info("WARNING: update_metrics received metrics_n_samples = 0 or < 0!!!!!!"+
                        f"The metric_stat is :{metric_stat}")
            raise NotImplementedError
            return
        self.metrics_n_samples += metrics_n_samples
        for thing in self.things_to_metric:
            self.stat[thing].update(metric_stat[thing], metrics_n_samples)

    def get_thing_tracker(self, thing):
        return self.local_recorder.tracker_dict[thing]


    def update_local_record(self, thing, client_index=None, server_index=None,
            summary_n_samples=None, time_info=None, args=None, **kwargs):
        key_info = {}
        if time_info is None:
            key_info.update(self.get_local_time_info_to_upload())
        else: 
            key_info.update(time_info)
        if client_index is not None:
            assert server_index is None
            key_info["client_index"] = client_index
        if server_index is not None:
            key_info["server_index"] = server_index
        self.local_recorder.update_record(
            key_info,
            thing,
            summary_n_samples,
            args,
            **kwargs
        )

    # =====================================================================
    def update_time_stamp(self, absolute_epoch, inner_iteration):
        """
        Here the epoch should be a global epoch, to identify infomations.
        And the iteration is accounted in each epoch.
        """
        self.absolute_epoch = absolute_epoch
        self.inner_iteration = inner_iteration

    def get_global_time_info_to_upload(self):
        time_info = {
            "global_comm_round": self.timer.global_comm_round_idx, 
            "epoch": self.timer.global_outer_epoch_idx, 
            "iteration": self.timer.global_outer_iter_idx
        }
        return time_info

    def get_local_time_info_to_upload(self):
        if self.timer.role == 'client':
            time_info = {
                "global_comm_round": self.timer.global_comm_round_idx,
                "local_outer_epoch": self.timer.local_outer_epoch_idx,
                "local_outer_iter": self.timer.local_outer_iter_idx
            }
        elif self.timer.role == 'server':
            time_info = {
                "global_comm_round": self.timer.global_comm_round_idx,
                "local_outer_epoch": self.timer.global_outer_epoch_idx,
                "local_outer_iter": self.timer.global_outer_iter_idx
            }
        else:
            raise NotImplementedError
        return time_info


    def get_epoch(self):
        return self.absolute_epoch

    def get_iteration(self):
        return self.inner_iteration

    # def add_time_info_local(self, client_index, time_info):
    #     self.time_info_locals[client_index] = time_info

    # def get_time_info_local(self, client_index):
    #     return self.time_info_locals[client_index]

    # def get_epoch_local(self, client_index):
    #     return self.time_info_locals[client_index]["epoch"]

    # def get_iteration_local(self, client_index):
    #     return self.time_info_locals[client_index]["iteration"]
    # =====================================================================

    #  operations with outside trackers
    # the summary_global will be transmitted to wandb, or other usage.
    # the record will be saved, so all things will not be reset.
    # =====================================================================
    def add_summary_info_local(self, client_index, local_summary_info, summary_n_samples=1):
        """ Now both dict and int are the same codes."""
        if type(summary_n_samples) == dict:
            self.summary_info_locals[client_index].set_info(local_summary_info)
            self.summary_info_locals[client_index].set_summary_n_samples(summary_n_samples)
            self.summary_global.update(local_summary_info, summary_n_samples)
        else:
            logging.warning(" WARNING: The summary_n_samples should be indicated!!!")
            self.summary_info_locals[client_index].set_info(local_summary_info)
            self.summary_info_locals[client_index].set_summary_n_samples(summary_n_samples)
            self.summary_global.update(local_summary_info, summary_n_samples)

    def get_summary_info_local(self, client_index):
        things = self.summary_info_locals[client_index].get_things()
        summary = self.summary_info_locals[client_index].get_info()
        summary_n_samples = self.summary_info_locals[client_index].get_summary_n_samples()
        return things, summary, summary_n_samples


    def add_summary_info_server(self, server_summary_info, summary_n_samples=1):
        if type(summary_n_samples) == dict:
            self.summary_info_server.set_info(server_summary_info)
            self.summary_info_server.set_summary_n_samples(summary_n_samples)
            # summary_global is used for averaging clients summary.
            # self.summary_global.update(server_summary_info, summary_n_samples)
        else:
            logging.warning(" WARNING: The summary_n_samples should be indicated!!!")
            self.summary_info_server.set_info(server_summary_info)
            self.summary_info_server.set_summary_n_samples(summary_n_samples)
            # self.summary_global.update(server_summary_info, summary_n_samples)


    def get_summary_info_server(self, server_index=0):
        things = self.summary_info_server.get_things()
        summary = self.summary_info_server.get_info()
        summary_n_samples = self.summary_info_server.get_summary_n_samples()
        return things, summary, summary_n_samples


    def get_summary_info_global(self):
        # things = self.summary_global.things
        # _, summary, _ = self.summary_global.get_summary()
        # summary_n_samples = self.summary_global.get_summary_n_samples_dict()
        return self.summary_global.get_summary()
    # =====================================================================

    # =====================================================================
    def add_to_global_recorder(self, record, summary_n_samples=None):
        self.global_recorder.add_record(record, summary_n_samples)

    def get_global_recorder(self):
        return self.global_recorder.get_record()
    # =====================================================================

    # =====================================================================
    def get_local_recorder(self):
        return self.local_recorder.get_record()

    def add_to_local_recorder(self):
        logging.error("ERROR!!  temp recorder now is only updated by update_record")
        raise NotImplementedError
    # =====================================================================

    # =====================================================================
    def get_summary_local_recorder(self):
        things, summary_info, summary_n_samples_dict = self.local_recorder.get_summary()
        return things, summary_info, summary_n_samples_dict

    def get_summary_global_recorder(self):
        things, summary_info, summary_n_samples_dict = self.global_recorder.get_summary()
        return things, summary_info, summary_n_samples_dict
    # =====================================================================


    # =====================================================================
    def get_things_summary_global_recorder(self, things):
        summary_info = {}
        summary_n_samples_dict = {}
        for thing in things:
            summary, summary_n_samples = self.global_recorder.get_thing_summary(thing)
            summary_info[thing] = summary
            summary_n_samples_dict[thing] = summary_n_samples
        return summary_info, summary_n_samples_dict

    def get_things_summary_local_recorder(self, things):
        summary_info = {}
        summary_n_samples_dict = {}
        for thing in things:
            summary, summary_n_samples = self.local_recorder.get_thing_summary(thing)
            summary_info[thing] = summary
            summary_n_samples_dict[thing] = summary_n_samples
        return summary_info, summary_n_samples_dict

    def get_things_summary_global(self, things):
        summary_info = {}
        summary_n_samples_dict = {}
        for thing in things:
            summary, summary_n_samples = self.summary_global.get_thing_summary(thing)
            summary_info[thing] = summary
            summary_n_samples_dict[thing] = summary_n_samples
        return summary_info, summary_n_samples_dict

    def get_things_summary_local(self, things, client_index):
        summary_info = {}
        summary_n_samples_dict = {}
        for thing in things:
            summary_info[thing] = self.summary_info_locals[client_index].get_thing_info(thing)
            summary_n_samples_dict[thing] = self.summary_info_locals[client_index].get_thing_summary_n_samples(thing)
        return summary_info, summary_n_samples_dict
    # =====================================================================

    def get_metric_info(self, metrics=None, **kwargs):
        if metrics is None or metrics.get_metric_info_func is None:
            metric_info = dict((name, val.avg) for name, val in self.stat.items())
        else:
            metric_info = metrics.get_metric_info_func(self.stat, **kwargs)
        return metric_info

    def get_local_info(self, metrics=None, **kargs):
        info = {}
        info['summary_n_samples'] = self.summary_n_samples
        metric_info = self.get_metric_info(metrics=metrics, **kargs)
        info["metric_info"] = metric_info
        time_info = self.get_local_time_info_to_upload()
        info["time_info"] = time_info
        things, summary_info, summary_n_samples_dict = self.get_summary_local_recorder()
        info["record_summary_info"] = summary_info
        return info

    def get_global_info(self, metrics=None, **kargs):
        info = {}
        info['summary_n_samples'] = self.summary_n_samples
        metric_info = self.get_metric_info(metrics=metrics, **kargs)
        info["metric_info"] = metric_info
        time_info = self.get_global_time_info_to_upload()
        info["time_info"] = time_info
        things, summary_info, summary_n_samples_dict = self.get_summary_global_recorder()
        info["record_summary_info"] = summary_info
        return info

    #  Message send and receive between workers or servers.
    # =====================================================================
    def encode_local_info(self, client_index, if_reset, metrics=None, **kwargs):
        tracker_info = {}
        tracker_info["client_index"] = client_index
        record = self.get_local_recorder()
        things, summary_info_local, summary_n_samples_dict = self.get_summary_local_recorder()
        if self.args.record_dataframe:
            tracker_info["record"] = record
            # logging.info("Local record dataframe is {}".format(
            #     tracker_info["record"]
            # ))
            # logging.info("Head {}".format(
            #     tracker_info["record"].head()
            # ))
        else:
            tracker_info["record"] = {}
        tracker_info["things"] = things
        tracker_info["summary_info_local"] = summary_info_local
        tracker_info["summary_n_samples"] = summary_n_samples_dict
        tracker_info["time_info"] = self.get_local_time_info_to_upload()
        metric_info = self.get_metric_info(metrics=metrics, **kwargs)
        tracker_info["metrics_n_samples"] = self.metrics_n_samples
        tracker_info["metric_info"] = metric_info
        if if_reset is True:
            self.reset()
        return tracker_info
        # return deepcopy(tracker_info)

    def decode_local_info(self, client_index, tracker_info, **kwargs):
        if client_index != tracker_info["client_index"]:
            logging.info("client_index: {}, get client_index from tracker_info: {}".format(
                client_index, tracker_info["client_index"]
            ))
        summary_n_samples = tracker_info["summary_n_samples"]
        if self.args.record_dataframe:
            # logging.info("Recevie Local record dataframe is {}".format(
            #     tracker_info["record"]
            # ))
            # logging.info("Recevie Head {}".format(
            #     tracker_info["record"].head()
            # ))
            self.add_to_global_recorder(tracker_info["record"], summary_n_samples)
        self.add_summary_info_local(client_index, tracker_info["summary_info_local"], summary_n_samples)
        # self.add_time_info_local(client_index, tracker_info["time_info"])

        metrics_n_samples = tracker_info["metrics_n_samples"]
        self.update_metrics(tracker_info["metric_info"], metrics_n_samples)
    # =====================================================================


    #  Upload tracker info to wandb.
    # =====================================================================
    def upload_metric_and_suammry_info_to_wandb(self, if_reset, metrics=None, **kwargs):
        upload_info = {}
        metric_info = self.get_metric_info(metrics=metrics, **kwargs)
        upload_info.update(metric_info)

        for client_index in self.args.wandb_upload_client_list:
            if client_index in self.summary_info_locals:
                pass
            else:
                continue
            things, summary_info_local, summary_n_samples = self.get_summary_info_local(client_index)
            for key, value in summary_info_local.items():
                upload_info[key+"/client"+str(client_index)] = value
            # summary_info_local_new = {}
            # for key, value in summary_info_local.items():
            #     summary_info_local_new[key+"/client"+str(client_index)] = value
            # upload_info.update(summary_info_local_new)

        # this local recorder is used by server. 
        # Clients will not use this.
        things, summary_info_server, summary_n_samples_dict = self.get_summary_local_recorder()
        for key, value in summary_info_server.items():
            upload_info[key+"/server"+str(0)] = value

        # things, summary_info_server, summary_n_samples = self.get_summary_info_server()
        # for key, value in summary_info_server.items():
        #     upload_info[key+"/server"+str(0)] = value

        _, summary_info_global, _ = self.get_summary_info_global()
        # logging.info("summary_info_global: {}".format(summary_info_global))
        for key, value in summary_info_global.items():
            upload_info[key+"/global"] = value

        time_info = self.get_global_time_info_to_upload()

        prefix = "Mode: {}, Algorithm-{}, lr-{}, dataset-{}, model-{} ".format(
            self.mode,
            self.args.algorithm,
            self.args.lr,
            self.args.dataset,
            self.args.model
        )

        time_str = "Global comm round: {}, Global total epochs: {}, \
            Global Inter iterations: {}...".format(
                time_info["global_comm_round"],
                time_info["epoch"],
                time_info["iteration"],
            )

        # logging.info( + 'Train: ' + metrics.str_fn(metric_info))
        logging.info(prefix+time_str+str(upload_info))

        if self.args.wandb_record:
            wandb_log(prefix=self.mode, sp_values=upload_info, com_values=time_info,
                    update_summary=True, wandb_summary_dict=self.wandb_summary_dict)
        if if_reset is True:
            self.reset()

    def upload_record_to_wandb(self):
        if self.args.wandb_save_record_dataframe and self.args.wandb_record:
            assert self.args.record_dataframe is True
            persistent_record = self.global_recorder.get_record()
            save_path = os.path.join(wandb.run.dir, self.mode + "persistent_record.csv")
            persistent_record.to_csv(save_path)
            logging.info("Saving global record ...........  save path is :{}".format(
                save_path
            ))

            wandb.save(save_path)
    # =====================================================================



# def get_tracker_info(train_tracker, test_tracker, if_reset, metrics=None):
#     train_tracker_info = train_tracker.get_local_record_and_summary_to_send(metrics)
#     test_tracker_info = test_tracker.get_local_record_and_summary_to_send(metrics)
#     if if_reset:
#         train_tracker.reset()
#         test_tracker.reset()
#     else:
#         logging.info("WARNING: train_tracker and test_tracker are not reset!!!")
#     return train_tracker_info, test_tracker_info



# def get_metric_info(train_tracker, test_tracker, if_reset, metrics=None):
#     train_metric_info = train_tracker(metrics)
#     test_metric_info = test_tracker(metrics)
#     if if_reset:
#         train_tracker.reset()
#         test_tracker.reset()
#     else:
#         logging.info("WARNING: train_tracker and test_tracker are not reset!!!")
#     return train_metric_info, test_metric_info


# class RuntimeTracker(object):
#     """Tracking the runtime stat for local training."""

#     # def __init__(self, metrics_to_track=["top1"], on_cuda=True):
#     def __init__(self, things_to_track=["loss"], on_cuda=True, id=None):
#         self.things_to_track = things_to_track
#         self.on_cuda = on_cuda
#         self.summary_n_samples = 0
#         self.time_stamp = 0
#         self.id = id
#         self.stat = None
#         self.reset()

#     def reset(self):
#         self.stat = dict((name, AverageMeter()) for name in self.things_to_track)
#         self.summary_n_samples = 0

#     # def evaluate_global_metric(self, metric):
#     #     return global_average(
#     #         self.stat[metric].sum, self.stat[metric].count, on_cuda=self.on_cuda
#     #     ).item()

#     # def evaluate_global_metrics(self):
#     #     return [self.evaluate_global_metric(metric) for metric in self.metrics_to_track]

#     def get_metrics_performance(self):
#         return [self.stat[thing].avg for thing in self.things_to_track]

#     def update_metrics(self, metric_stat, summary_n_samples):
#         if summary_n_samples == 0 or summary_n_samples < 0:
#             logging.info("WARNING: update_metrics received summary_n_samples = 0 or < 0!!!!!!")
#             return
#         self.summary_n_samples += summary_n_samples
#         for thing in self.things_to_track:
#             self.stat[thing].update(metric_stat[thing], summary_n_samples)

#     def update_time_stamp(self, time_stamp):
#         self.time_stamp = time_stamp

#     def get_metric_info(self):
#         metric_info = dict((name, val.avg) for name, val in self.stat.items())
#         metric_info['summary_n_samples'] = self.summary_n_samples
#         metric_info['time_stamp'] = self.time_stamp
#         return metric_info

#     def __call__(self, metrics=None, **kargs):
#         if metrics is None or metrics.get_metric_info_func is None:
#             metric_info = dict((name, val.avg) for name, val in self.stat.items())
#             metric_info['summary_n_samples'] = self.summary_n_samples
#             metric_info['time_stamp'] = self.time_stamp
#         else:
#             metric_info = metrics.get_metric_info_func(self.stat, **kargs)
#             metric_info['summary_n_samples'] = self.summary_n_samples
#             metric_info['time_stamp'] = self.time_stamp

#         return metric_info


# class BestPerf(object):
#     def __init__(self, best_perf=None, larger_is_better=True):
#         self.best_perf = best_perf
#         self.cur_perf = None
#         self.best_perf_locs = []
#         self.larger_is_better = larger_is_better

#         # define meter
#         self._define_meter()

#     def _define_meter(self):
#         self.meter = MaxMeter() if self.larger_is_better else MinMeter()

#     def update(self, perf, perf_location):
#         self.is_best = self.meter.update(perf)
#         self.cur_perf = perf

#         if self.is_best:
#             self.best_perf = perf
#             self.best_perf_locs += [perf_location]

#     def get_best_perf_loc(self):
#         return self.best_perf_locs[-1] if len(self.best_perf_locs) != 0 else None




# def get_metric_info(train_tracker, test_tracker, time_stamp, if_reset, metrics=None):
#     train_tracker.update_time_stamp(time_stamp=time_stamp)
#     train_metric_info = train_tracker(metrics)
#     test_tracker.update_time_stamp(time_stamp=time_stamp)
#     test_metric_info = test_tracker(metrics)

#     if if_reset:
#         train_tracker.reset()
#         test_tracker.reset()
#     else:
#         logging.info("WARNING: train_tracker and test_tracker are not reset!!!")
#     return train_metric_info, test_metric_info


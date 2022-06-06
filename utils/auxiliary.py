def get_fullname(o):
    """get the full name of the class."""
    return '%s.%s' % (o.__module__, o.__class__.__name__)


class dict2obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a,
                        [dict2obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, dict2obj(b) if isinstance(b, dict) else b)


def check_test_frequency(args, timer):
    return timer.global_outer_epoch_idx % args.frequency_of_the_test == 0 or \
        timer.global_outer_epoch_idx == args.max_epochs - 1


def check_and_test(
        args,
        timer, trainer, test_global, device,
        train_tracker, test_tracker, metrics
    ):

    if check_test_frequency(args, timer):
        epoch = timer.global_outer_epoch_idx
        trainer.test(test_global, device, args, epoch, test_tracker, metrics)
        train_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=metrics)
        test_tracker.upload_metric_and_suammry_info_to_wandb(if_reset=True, metrics=metrics)

    else:
        train_tracker.reset()
        test_tracker.reset()






























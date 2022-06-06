from .model_dif_track import model_dif_tracker
from .generator_track import generator_tracker

def create_trackers(args, **kwargs):
    """
        create the recorder to record something during training.
    """
    recorder_dict = {}
    if args.model_dif_track:
        recorder_dict["model_dif_track"] = model_dif_tracker(args)
    if args.VHL:
        recorder_dict["generator_track"] = generator_tracker(args)

    return recorder_dict

















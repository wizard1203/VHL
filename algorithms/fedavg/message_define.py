from algorithms.basePS.message_define import MyMessage


class MyMessage(MyMessage):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_INDEX = "client_index"
    MSG_ARG_KEY_GRAD_PARAMS = "grad_params"
    MSG_ARG_KEY_MODEL_INDEXES = "model_indexes"
    MSG_ARG_KEY_GRAD_INDEXES = "grad_indexes"

    # Indicate global and current communication round.
    MSG_ARG_KEY_GLOBAL_ROUND_INDEX = "global_round_idx"
    MSG_ARG_KEY_LOCAL_ROUND_INDEX = "local_round_idx"

    # Indicate current dataset sampling times.
    MSG_ARG_KEY_LOCAL_EPOCH_INDEX = "local_epoch_idx"
    MSG_ARG_KEY_LOCAL_ITER_INDEX = "local_iter_idx"
    MSG_ARG_KEY_LOCAL_TOTAL_ITER_INDEX = "local_total_iter_idx"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

    MSG_ARG_KEY_TRAIN_TRACKER_INFO = "train_metrics"
    MSG_ARG_KEY_TEST_TRACKER_INFO = "test_metrics"


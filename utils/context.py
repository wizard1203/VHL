import sys
import time
from contextlib import contextmanager
import threading

import traceback


@contextmanager
def raise_MPI_error(MPI):
    import logging
    logging.debug("Debugging, Enter the MPI catch error")
    try:
        yield
    except Exception as e:
        logging.info(e)
        traceback.print_exception(*sys.exc_info())
        # logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
        logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
        MPI.COMM_WORLD.Abort()
    # else:
    #     traceback.print_exception(*sys.exc_info())
    #     logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
    #     MPI.COMM_WORLD.Abort()

@contextmanager
def raise_error_without_process():
    import logging
    logging.debug("Debugging, Enter the MPI catch error")
    try:
        yield
    except Exception as e:
        logging.info(e)
        traceback.print_exception(*sys.exc_info())
        logging.info('traceback.format_exc():\n%s' % traceback.format_exc())

def raise_error_and_retry(max_try, time_gap, func, *args, **kargs):
    import logging
    logging.debug("Debugging, Enter raise_error_and_retry")
    sth = None
    for i in range(max_try):
        success_flag = False
        try:
            sth = func(*args, **kargs)
            success_flag = True
        except Exception as e:
            success_flag = False
            logging.info(e)
            traceback.print_exception(*sys.exc_info())
            logging.info('traceback.format_exc():\n%s' % traceback.format_exc())
        if success_flag:
            logging.info("{}-th attempt to do func: {}, success!!!".format(i, str(func)))
            break
        else:
            logging.info("{}-th attempt to do func: {}, fail!!!, waiting {} seconds, retry....".format(
                i, str(func), time_gap))
            time.sleep(time_gap)
    return sth


@contextmanager
def get_lock(lock: threading.Lock()):
    lock.acquire()
    yield
    if lock.locked():
        lock.release()

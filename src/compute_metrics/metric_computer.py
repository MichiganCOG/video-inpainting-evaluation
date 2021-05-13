from abc import ABC, abstractmethod
import traceback
from multiprocessing import Process

from ..common_util.misc import no_stdout_stderr


class MetricComputer(Process, ABC):

    def __init__(self, opts, queue, worker_log_file):
        super().__init__()
        self.opts = opts
        self.queue = queue
        self.worker_log_file = worker_log_file


    @abstractmethod
    def compute_metric(self):
        raise NotImplementedError


    def run(self):
        with no_stdout_stderr(self.worker_log_file):
            try:
                self.compute_metric()
            except Exception as e:
                traceback.print_exc()
                self.send_exception_msg(e)


    def send_work_count_msg(self, work_count):
        self.queue.put(('count', self.name, work_count))


    def send_update_msg(self, update_count):
        self.queue.put(('update', self.name, update_count))


    def send_result_msg(self, result):
        self.queue.put(('result', self.name, result))


    def send_exception_msg(self, exception):
        self.queue.put(('exception', self.name, exception))

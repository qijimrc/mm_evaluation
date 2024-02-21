import torch
import logging

class Logger(object):
    _DEFAULT_NAME = "default"

    def __init__(self):
        self.eval_model_name = self._DEFAULT_NAME
        self.eval_task_name = self._DEFAULT_NAME

        self.logger = self._configure_logging()

    def _configure_logging(self):
        logger = logging.getLogger("mmeval")
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        return logger
        
    def set_model_name(self, model_name):
        self.eval_model_name = model_name

    def set_task_name(self, task_name):
        self.eval_task_name = task_name

    def reset_model_name(self):
        self.eval_model_name = self._DEFAULT_NAME

    def reset_task_name(self):
        self.eval_task_name = self._DEFAULT_NAME
        
    def get_model_name(self):
        return self.eval_model_name

    def get_task_name(self):
        return self.eval_task_name

    def __call__(self, print_str, add_sep_lines=False, sep_char='#', level=logging.INFO, flush=True):
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            print_items = [self.eval_model_name, self.eval_task_name]
            print_str = f"[{'-'.join(print_items)}] >> {print_str}"
            if add_sep_lines:
                print_str = f"\n{sep_char * 80}\n{print_str}\n{sep_char * 80}"
            self.logger.log(msg=print_str, level=level)
            if flush:
                self.logger.handlers[0].flush()

log = Logger()

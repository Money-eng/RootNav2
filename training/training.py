import logging
from run_training import train

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class LogFormatter(logging.Formatter):
    def format(self, record):
        self.datefmt='%H:%M:%S'
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            color = {
                logging.WARNING: 33,
                logging.ERROR: 31,
                logging.FATAL: 31,
                logging.DEBUG: 36
            }.get(record.levelno, 0)
            self._style._fmt = f"[%(asctime)s.%(msecs)03d] \033[{color}m%(levelname)s\033[0m: %(message)s"
        return super().format(record)

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(LogFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

class Args:
    def __init__(self):
        self.config = "./training/configs/root_train_cldice.yml" # redo with other config to remake the experiment
        
        self.output_example = True
        
        self.debug = False
        
        self.resume_iterations = False

if __name__ == "__main__":
    args = Args()
    print(f"Launching training with config: {args.config}")
    train(args)
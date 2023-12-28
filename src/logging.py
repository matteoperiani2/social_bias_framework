import wandb


class WandbLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WandbLogger, cls).__new__(cls)
        return cls._instance

    def init_wandb(self, **wandb_kwargs):
        self.__current_step = 0
        wb = wandb.init(**wandb_kwargs)
        return wb

    def log_step(self, **log_values):
        wandb.log(log_values, step=self.__current_step)

    def update_step(self, n: int):
        """Increment current step by n"""
        self.__current_step += n

    @property
    def current_step(self):
        return self.__current_step

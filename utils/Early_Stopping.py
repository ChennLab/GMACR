import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metric_max = -np.Inf  # 假设我们监控准确率 (越高越好)
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model):
        score = val_metric  # 对于准确率，分数就是准确率本身

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:  # 没有改善
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # 有改善
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            self.trace_func(f'Validation metric improved ({self.val_metric_max:.4f} --> {val_metric:.4f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_metric_max = val_metric

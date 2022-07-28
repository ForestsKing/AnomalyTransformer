import numpy as np
import torch


class EarlyStopping:
    def __init__(self, path, patience=3, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score_1 = None
        self.best_score_2 = None
        self.early_stop = False
        self.val_loss_1_min = np.Inf
        self.val_loss_2_min = np.Inf

    def __call__(self, val_loss_1, val_loss_2, model):
        score_1 = -val_loss_1
        score_2 = -val_loss_2

        if (self.best_score_1 is None) or (self.best_score_2 is None):
            self.best_score_1 = score_1
            self.best_score_2 = score_2
            self.save_checkpoint(val_loss_1, val_loss_2, model)

        elif (score_1 < self.best_score_1 + self.delta) or (score_2 < self.best_score_2 + self.delta):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score_1 = score_1
            self.best_score_2 = score_2
            self.save_checkpoint(val_loss_1, val_loss_2, model)
            self.counter = 0

    def save_checkpoint(self, val_loss_1, val_loss_2, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_1_min:.6f} --> {val_loss_1:.6f}) '
                  f'({self.val_loss_2_min:.6f} --> {val_loss_2:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.path)
        self.val_loss_1_min = val_loss_1
        self.val_loss_2_min = val_loss_2

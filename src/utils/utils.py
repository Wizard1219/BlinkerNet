import logging

import torch
import torch.nn as nn


def init_xavier(m):
    if type(m) == nn.Conv3d:
        nn.init.xavier_uniform_(m.weight)

def print_log(s):
    """Prints a string to the console and logs it to the log file."""
    print(s)
    logging.info(s)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
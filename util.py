import os
import torch
from datetime import datetime


def save_checkpoint(model, data_dir, epoch) :
    save_path = os.path.join(data_dir, datetime.now().strftime('%H-%M-%S') + '-' + str(epoch) + '_model.bin')
    torch.save(model.state_dict(), save_path)
    return save_path

import os
import shutil

import torch
from pyjeasy.file_utils import file_exists, make_dir_if_not_exists, move_file


def save_ckp(state, checkpoint_dir, best_model_dir, filename_pth='checkpoint.pth'):
    """
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
    """

    make_dir_if_not_exists(checkpoint_dir)
    make_dir_if_not_exists(best_model_dir)

    best_fpath = os.path.join(best_model_dir, 'best_model.pth')
    rename_f_path = os.path.join(best_model_dir, filename_pth)
    f_path = os.path.join(checkpoint_dir, filename_pth)
    if file_exists(best_fpath):
        os.rename(best_fpath, rename_f_path)
        move_file(rename_f_path, f_path)
    torch.save(state, best_fpath)
    # shutil.copyfile(f_path, best_fpath)
    # torch.save(state, f_path)
    # if is_best:


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    model = MyModel(*args, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ckp_path = "path/to/checkpoint/checkpoint.pt"
    model, optimizer, start_epoch = load_ckp(ckp_path, model, optimizer)
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

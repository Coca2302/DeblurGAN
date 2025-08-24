# utils.py
import torch

def load_generator_checkpoint(model, path, device):
    """
    Hỗ trợ 2 kiểu checkpoint:
    - state_dict trực tiếp
    - dict chứa 'generator' / 'generator_state_dict' / 'model_state_dict'
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        # tìm key phù hợp
        keys = ['generator', 'generator_state_dict', 'model_state_dict', 'state_dict']
        for k in keys:
            if k in ckpt:
                state = ckpt[k]
                model.load_state_dict(state)
                return model
        # nếu dict nhưng chính nó là một state_dict (kiểu torch.save(model.state_dict()))
        try:
            model.load_state_dict(ckpt)
            return model
        except Exception as e:
            raise RuntimeError(f"Không thể load checkpoint: {e}")
    else:
        # có thể file trực tiếp là state_dict
        try:
            model.load_state_dict(ckpt)
            return model
        except Exception as e:
            raise RuntimeError(f"Không thể load checkpoint (non-dict): {e}")

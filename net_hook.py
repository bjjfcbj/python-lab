import torch


def get_fea_by_hook(model):
    hook = HookTool()
    if model is None:
        raise UserWarning("model 不能为空")
    model.register_forward_hook(hook.hook_func)
    return hook


class HookTool:
    def __init__(self):
        self.fea = None

    def hook_func(self, module, fea_in, fea_out):
        self.fea = fea_out

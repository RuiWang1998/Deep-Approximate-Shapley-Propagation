import torch.nn as nn

from lightprobnets import contrib


def convert_2_lpdn(model:nn.Module, convert_weights:bool=True)->nn.Module:
    """
    Convert the model into a LPDN
    Conversion code skeleton from https://discuss.pytorch.org/t/how-can-i-replace-an-intermediate-layer-in-a-pre-trained-network/3586/7
    :param model: The model to convert
    :param convert_weights:
    :return: converted LPDN
    """
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_2_lpdn(module, convert_weights)

        else:
            if isinstance(module, nn.Conv2d):
                layer_old = module
                layer_new = contrib.adf.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                     module.padding, module.dilation, module.groups,
                                     module.bias is not None, module.padding_mode)
            elif isinstance(module, nn.Linear):
                layer_old = module
                layer_new = contrib.adf.Linear(module.in_features, module.out_features, module.bias is not None)
            elif isinstance(module, nn.ReLU):
                layer_old = module
                layer_new = contrib.adf.ReLU()
            else:
                raise NotImplementedError(f"Layer type {module} not supported")
            try:
                if convert_weights:
                    layer_new.weight = layer_old.weight
                    layer_new.bias = layer_old.bias
            except AttributeError:
                pass

            model._modules[name] = layer_new

    return model


class DASP(object):
    def __init__(self, model:nn.Module):
        self.model = model


    def _convert_to_lpdn(self, model:nn.Module):
        """
        Convert a generic torch module into lpdn model

        :param model:
        :type model:
        :return:
        :rtype:
        """
        pass


if __name__ == "__main__":
    convert_2_lpdn(nn.ModuleList([nn.ReLU(), nn.Linear(10,10), nn.ModuleList([nn.ReLU()])]))

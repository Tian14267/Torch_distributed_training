import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda", logger=None):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                #summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                
                summary[m_key]["output_shape"] = [[batch_size] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype_1 = torch.cuda.IntTensor
        dtype_2 = torch.cuda.FloatTensor
    else:
        dtype_1 = torch.IntTensor
        dtype_2 = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    #x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    x = []
    for i, in_size in enumerate(input_size):
        if i != len(input_size)-1:  ###  int
            x.append(torch.rand(2, *in_size).type(dtype_1))
        else:
            x.append(torch.rand(2, *in_size).type(dtype_2))


    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    def count_summary_layer_output_shape(summary_layer_output_shape):
        if isinstance(summary_layer_output_shape[0], int):
            all_num = np.prod(summary_layer_output_shape)
        else:
            all_num = np.sum([np.prod(in_tuple) for in_tuple in summary_layer_output_shape])

        return all_num



    #print("----------------------------------------------------------------")
    logger.info("----------------------------------------------------------------")
    line_new = "{:>25}  {:>35} {:>20}".format("Layer (type)", "Output Shape", "Param #")
    #print(line_new)
    logger.info(line_new)
    #print("================================================================")
    logger.info("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        summary_layer_output_shape = summary[layer]["output_shape"]
        summary_layer_nb_params = summary[layer]["nb_params"]
        layer_count = count_summary_layer_output_shape(summary_layer_output_shape)
        line_new = "{:>25}  {:>35} {:>20}".format(
            layer,
            str(summary_layer_output_shape),
            "{0:,}".format(summary_layer_nb_params),
        )
        total_params += summary_layer_nb_params
        total_output += layer_count
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary_layer_nb_params
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    #total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_input_size = abs(layer_count * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    #print("================================================================")
    logger.info('================================================================')
    #print("Total params: {0:,}".format(total_params))
    logger.info('Total params: {0:,}'.format(total_params))
    #print("Trainable params: {0:,}".format(trainable_params))
    logger.info("Trainable params: {0:,}".format(trainable_params))
    #print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    logger.info("Non-trainable params: {0:,}".format(total_params - trainable_params))
    #print("----------------------------------------------------------------")
    logger.info("----------------------------------------------------------------")
    #print("Input size (MB): %0.2f" % total_input_size)
    logger.info("Input size (MB): %0.2f" % total_input_size)
    #print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    logger.info("Forward/backward pass size (MB): %0.2f" % total_output_size)
    #print("Params size (MB): %0.2f" % total_params_size)
    logger.info("Params size (MB): %0.2f" % total_params_size)
    #print("Estimated Total Size (MB): %0.2f" % total_size)
    logger.info("Estimated Total Size (MB): %0.2f" % total_size)
    #print("----------------------------------------------------------------")
    logger.info("----------------------------------------------------------------")
    # return summary

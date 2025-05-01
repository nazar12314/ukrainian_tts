from collections import OrderedDict


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    return new_state_dict

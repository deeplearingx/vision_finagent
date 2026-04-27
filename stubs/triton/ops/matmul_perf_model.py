# Mock module to satisfy bitsandbytes import chain
def early_config_prune(configs, named_args, **kwargs):
    return configs

def estimate_matmul_time(**kwargs):
    return 0

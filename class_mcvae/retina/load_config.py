import yaml
from argparse import Namespace
from IPython import embed

def recursive_namespace(dd):
    '''
    Converts a (possibly nested) dictionary into a namespace.
    This allows for auto-completion
    '''
    for d in dd:
        has_any_dicts = False
        if isinstance(dd[d], dict):
            dd[d] = recursive_namespace(dd[d])
            has_any_dicts = True
    try:
      n =  Namespace(**dd)
      return n
    except:
      print(dd)


def load_config(yaml_config_file, args):
    
    
    with open(yaml_config_file) as config:
        config = yaml.safe_load(config)   
        # I am using a namespace instead of a dictionary mainly because it enables auto-completion
    config = recursive_namespace(config)
    if args.w_kld is not None:
        config.w_kld = args.w_kld

    if args.latent_dim is not None:
        config.latent_dim = args.latent_dim

    if args.batch_size is not None:
        config.optimizer.batch_size = args.batch_size

#### FUNDUS
    config.network_architecture_fundus.convolution.parameters.channels = \
    [int(x) for x in config.network_architecture_fundus.convolution.parameters.channels.split()]
    config.network_architecture_fundus.convolution.parameters.padding = \
    [int(x) for x in config.network_architecture_fundus.convolution.parameters.padding.split()]
    config.network_architecture_fundus.convolution.parameters.stride = \
    [int(x) for x in config.network_architecture_fundus.convolution.parameters.stride.split()] 
    config.network_architecture_fundus.convolution.parameters.kernel_size = \
    [int(x) for x in config.network_architecture_fundus.convolution.parameters.kernel_size.split()]
    config.input_dim_fundus = \
    [int(x) for x in config.input_dim_fundus.split()]

### OCT
    config.network_architecture_oct.convolution.parameters.channels = \
    [int(x) for x in config.network_architecture_oct.convolution.parameters.channels.split()]
    config.network_architecture_oct.convolution.parameters.padding = \
    [int(x) for x in config.network_architecture_oct.convolution.parameters.padding.split()]
    config.network_architecture_oct.convolution.parameters.stride = \
    [int(x) for x in config.network_architecture_oct.convolution.parameters.stride.split()]
    config.network_architecture_oct.convolution.parameters.kernel_size = \
    [int(x) for x in config.network_architecture_oct.convolution.parameters.kernel_size.split()]
    config.input_dim_oct = \
    [int(x) for x in config.input_dim_oct.split()]
    
    return config

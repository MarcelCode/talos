def parallel_gpu_jobs(allow_growth=True, fraction=.5):
    '''Sets the max used memory as a fraction for tensorflow
    backend

    allow_growth :: True of False

    fraction :: a float value (e.g. 0.5 means 4gb out of 8gb)

    '''

    from tensorflow.compat.v1 import GPUOptions, ConfigProto, Session
    from tensorflow.compat.v1 import keras as K

    gpu_options = GPUOptions(allow_growth=allow_growth,
                             per_process_gpu_memory_fraction=fraction)
    config = ConfigProto(gpu_options=gpu_options)
    session = Session(config=config)
    K.backend.set_session(session)


def multi_gpu(model, gpus=None, cpu_merge=True, cpu_relocation=False):
    '''Takes as input the model, and returns a model
    based on the number of GPUs available on the machine
    or alternatively the 'gpus' user input.

    NOTE: this needs to be used before model.compile() in the
    model inputted to Scan in the form:

    from talos.utils.gpu_utils import multi_gpu
    model = multi_gpu(model)

    '''

    from tensorflow.keras.utils import multi_gpu_model

    return multi_gpu_model(model,
                           gpus=gpus,
                           cpu_merge=cpu_merge,
                           cpu_relocation=cpu_relocation)


def force_cpu():
    '''Force CPU on a GPU system
    '''

    from tensorflow.compat.v1 import ConfigProto, Session
    from tensorflow.compat.v1 import keras as K

    config = ConfigProto(device_count={'GPU': 0})
    session = Session(config=config)
    K.backend.set_session(session)

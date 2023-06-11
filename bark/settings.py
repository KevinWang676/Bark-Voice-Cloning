import os

def initenv(args):
    os.environ['SUNO_USE_SMALL_MODELS'] = str("-smallmodels" in args)
    os.environ['BARK_FORCE_CPU'] = str("-forcecpu" in args)
    os.environ['SUNO_ENABLE_MPS'] = str("-enablemps" in args)
    os.environ['SUNO_OFFLOAD_CPU'] = str("-offloadcpu" in args)

###################################################
#
# general utility tools for agrument checking, etc
#
###################################################


def check_args(kwargs, required_args):
    ''' Checks if the kwargs contain ALL of the required_args keys '''
    if not isinstance(required_args, (list,tuple)):
        required_args = [required_args]
    pres = [True if a in kwargs.keys() else False for a in required_args]
    missing = [required_args[i] for i,b in enumerate(pres) if not b]

    return all(pres), missing


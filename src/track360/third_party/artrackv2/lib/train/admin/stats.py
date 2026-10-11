"""Minimal classes needed to unpickle official ARTrack archives.

The runtime only consumes the ``net`` state dictionary.  These metadata shells
avoid importing the full training stack when loading a trusted local checkpoint.
"""

class StatValue:
    pass

class AverageMeter:
    pass

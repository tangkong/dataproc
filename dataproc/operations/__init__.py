"""
Operations submodule

Sets out structure for atomic operations with basic type hinting.  

Based loosely off of Xi-cam's operation plugin branch
"""

class Operation(object):
    def __init__(self, func):
        self._func = func
        self.name = getattr(func, 'name', getattr(func, '__name__', None))
        if self.name is None:
            raise NameError('provided operation is unnamed')
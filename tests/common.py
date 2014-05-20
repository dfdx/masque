
import os

def interactive(fn):
    def wrapped(*args, **kwargs):
        if os.environ.get('INTERACTIVE'):
            return fn(*args, **kwargs)
        else:             
            print 'Ignoring interactive function: %s' % fn.__name__
    return wrapped
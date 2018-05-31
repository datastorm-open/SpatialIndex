import functools
import inspect
import multiprocessing

import toolz

# Add checks for n_jobs and chunk_size in arguments to fnc.
# Possibly use parameters[key].kind to get the kind of param.
# Using name(s) to pmap on which variables. By default, None means first arg.

def pmap(fnc, n_jobs=1, chunk_size=10000):
    @functools.wraps(fnc)
    def wrapper(iterable, *args,
                n_jobs=n_jobs, chunk_size=chunk_size, **kwargs):
        # Vectorize
        params = list(inspect.signature(fnc).parameters.keys())
        kwgs = dict(zip(params[1:], args))
        kwgs.update(kwargs)
        task = toolz.compose(
            list, toolz.curry(map)(toolz.partial(fnc, **kwgs)))
        # Parallelize
        if n_jobs == 1:
            return task(iterable)
        chunks = list(toolz.partition_all(chunk_size, iterable))
        with multiprocessing.Pool(n_jobs) as pool:
            res = pool.map(task, chunks)
        return [y for x in res for y in x]
    return wrapper

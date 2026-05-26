from gsn.utils.ops import (   # all safe now (TF imported lazily inside fns)
    add_self_loops_local,
    gather_src,
    gather_dst,
    aggregate,
)

__all__ = ["gather_src", "gather_dst", "aggregate", "add_self_loops_local"]

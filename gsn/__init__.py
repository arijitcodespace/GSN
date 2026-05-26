from gsn.snapshot import Snapshot

__all__ = ["Snapshot"]

def __getattr__(name):
    if name == "DenseStateTable":
        from gsn.state.table import DenseStateTable
        return DenseStateTable
    raise AttributeError(f"module 'gsn' has no attribute {name!r}")

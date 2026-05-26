from tensorflow.keras import layers

import contextvars
from contextlib import ContextDecorator

_pinned_num_args = contextvars.ContextVar("pinned_num_args")


class PinArgs(ContextDecorator):
    def __init__(self, num_args: int):
        num_args = int(num_args)
        if num_args <= 0:
            raise ValueError("num_args must be positive")

        self.num_args = num_args
        self._token = None

    def __enter__(self):
        self._token = _pinned_num_args.set(self.num_args)
        return self

    def __exit__(self, exc_type, exc, tb):
        _pinned_num_args.reset(self._token)
        return False

    @staticmethod
    def current() -> int:
        try:
            return _pinned_num_args.get()
        except LookupError as e:
            raise RuntimeError(
                                "num_call_args was not provided. "
                                "Either pass num_call_args=... explicitly or construct this layer "
                                "inside `with PinArgs(num_args=...):`."
                              ) from e


class GraphLayerBackbone(layers.Layer):
    def __init__(self, num_call_args = None, **kwargs):
        super().__init__(**kwargs)
        
        if num_call_args is None:
            num_call_args = PinArgs.current()
        
        self.num_call_args = int(num_call_args)
        
        
    def build(self, input_shape):
        super().build(input_shape)
    
    def __call__(self, *args, **kwargs):
        if len(args) != self.num_call_args:
            raise RuntimeError(f"Expected {self.num_call_args} positional arguments. "
                               f"Found {len(args)}. Either pass the rest as keyword "
                               f"arguments or reinstantiate the class with "
                               f"appropriate `num_call_args`.")
        
        return super().__call__(*args, **kwargs)
    
    
    def call(self, *args, **kwargs):
        raise NotImplementedError(f"Sub-classes are expected to override the `call()` method.")
    
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_call_args": self.num_call_args})
        return cfg
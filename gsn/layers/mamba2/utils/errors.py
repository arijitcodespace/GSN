class ShapeError(Exception):
    def __init__(self, message, **kwargs):
        super().__init__(message, **kwargs)
def get_type_of_arguments(*args, **kwargs):
    pos_args = []
    kw_args = []
    for arg in args:
        pos_args.append(type(arg))
    for name, val in kwargs.items():
        kw_args.append(dict(name = type(val)))
        
    return pos_args, kw_args
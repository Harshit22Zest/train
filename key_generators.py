pattern = '{}'


def _make_key(*args, **kwargs):
    """
    simple python join with lodash separator
    :param args: :type any iterable
    :return: string separated by lodash  #  '_'
    """
    separator = kwargs.get('separator', '_')
    return separator.join(args)


def make_key(*args, **kwargs):
    """:_make_key
    Handles when arg value is NoneType(regular formatting not possible) also.
    :param args: :type any iterable
    :return: :return: _make_key
    """
    un_formatted = _make_key(*([pattern] * len(args)), **kwargs)  # to avoid NoneType exception
    return un_formatted.format(*args)

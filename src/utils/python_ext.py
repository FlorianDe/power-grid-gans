class FinalClass(type):
    """
    We can emulate Java-Like final classes with this.
    """

    def __new__(cls, name, bases, classdict):
        for b in bases:
            if isinstance(b, FinalClass):
                raise TypeError("type '{0}' is not an acceptable base type".format(b.__name__))
        return type.__new__(cls, name, bases, dict(classdict))

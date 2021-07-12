import numpy

DATETIME_CONDITIONAL_VECTOR_SIZE = 14

def dates_to_conditional_vectors(months:  numpy.ndarray, days: numpy.ndarray, hours: numpy.ndarray):
    converted_dates = (months << 10) + (days << 5) + hours
    return [[int(x) for x in list('{0:0{bits}b}'.format(d, bits=DATETIME_CONDITIONAL_VECTOR_SIZE))] for d in converted_dates]
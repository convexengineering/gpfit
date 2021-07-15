"Implements functions for raw fit printing from params"
from numpy import exp


# pylint: disable=invalid-name
def print_isma(A, B, alpha, d, K):
    """prints ISMA fit from params"""

    print("ISMA fit from params")
    string_list = [None]*K
    print_string = "1 = "
    for k in range(K):
        if k > 0:
            print(print_string)
            print_string = "    + "
        print_string += "({0:.6g}/w**{1:.6g})".format(exp(alpha[k]*B[k]), alpha[k])
        for i in range(d):
            print_string += " * (u_{0:d})**{1:.6g}".format(
                i + 1, alpha[k]*A[d*k + i]
            )
        string_list[k] = print_string
    print(print_string)

    return string_list


# pylint: disable=invalid-name
def print_sma(A, B, alpha, d, K):
    """prints SMA fit from params"""

    print("SMA fit from params")
    string_list = [None]*K
    print_string = "w**{0:.6g} = ".format(alpha)
    for k in range(K):
        if k > 0:
            print(print_string)
            print_string = "    + "
        print_string += "{0:.6g}".format(exp(alpha*B[k]))
        for i in range(d):
            print_string += " * (u_{0:d})**{1:.6g}".format(i + 1, alpha*A[d*k + i])
        string_list[k] = print_string
    print(print_string)

    return string_list


# pylint: disable=invalid-name
def print_ma(A, B, d, K):
    """prints MA fit from params"""

    print("MA fit from params")
    string_list = [None]*K
    for k in range(K):
        print_string = "w = {0:.6g}".format(exp(B[k]))
        for i in range(d):
            print_string += " * (u_{0:d})**{1:.6g}".format(i + 1, A[d*k + i])
        string_list[k] = print_string
        print(print_string)

    return string_list

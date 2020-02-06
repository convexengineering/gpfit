"Implements functions for raw fit printing from params"
from __future__ import print_function
from builtins import range
from numpy import exp


# pylint: disable=invalid-name
def print_ISMA(A, B, alpha, d, K):
    "prints ISMA fit from params"
    print("ISMA fit from params")
    stringList = [None]*K

    printString = '1 = '
    for k in range(K):
        if k > 0:
            print(printString)
            printString = '    + '

        printString += '({0:.6g}/w**{1:.6g})'.format(exp(alpha[k] * B[k]), alpha[k])

        for i in range(d):
            printString += ' * (u_{0:d})**{1:.6g}'.format(i+1, alpha[k] * A[d*k + i])

        stringList[k] = printString

    print(printString)
    return stringList


# pylint: disable=invalid-name
def print_SMA(A, B, alpha, d, K):
    "prints SMA fit from params"
    print("SMA fit from params")
    stringList = [None]*K

    printString = 'w**{0:.6g} = '.format(alpha)
    for k in range(K):
        if k > 0:
            print(printString)
            printString = '    + '

        printString += '{0:.6g}'.format(exp(alpha * B[k]))

        for i in range(d):
            printString += ' * (u_{0:d})**{1:.6g}'.format(i+1, alpha * A[d*k + i])

        stringList[k] = printString

    print(printString)
    return stringList


# pylint: disable=invalid-name
def print_MA(A, B, d, K):
    "prints MA fit from params"
    print("MA fit from params")
    stringList = [None]*K

    for k in range(K):
        printString = 'w = {0:.6g}'.format(exp(B[k]))

        for i in range(d):
            printString += ' * (u_{0:d})**{1:.6g}'.format(i+1, A[d*k + i])

        stringList[k] = printString
        print(printString)

    return stringList

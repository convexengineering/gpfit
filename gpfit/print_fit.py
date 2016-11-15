from numpy import arange, exp, array

def print_ISMA(A, B, alpha, d, K):

    stringList = [None]*K

    printString = '1 = '
    for k in range(K):
        if k > 0:
            print printString
            printString = '    + '

        printString += '({0:.3g}/w**{1:.3g})'.format(exp(alpha[k] * B[k]), alpha[k])

        for i in range(d):
            printString += ' * (u_{0:d})**{1:.3g}'.format(i+1, alpha[k] * A[d*k + i])

        stringList[k] = printString

    print printString
    return stringList


def print_SMA(A, B, alpha, d, K):

    stringList = [None]*K

    printString = 'w**{0:.6g} = '.format(alpha)
    for k in range(K):
        if k > 0:
            print printString
            printString = '    + '

        printString += '{0:.6g}'.format(exp(alpha * B[k]))

        for i in range(d):
            printString += ' * (u_{0:d})**{1:.6g}'.format(i+1, alpha * A[d*k + i])

        stringList[k] = printString

    print printString
    return stringList


def print_MA(A, B, d, K):
    '''
    Print set of K monomial inequality constraints
    '''

    stringList = [None]*K

    for k in range(K):
        printString = 'w = {0:.3g}'.format(exp(B[k]))

        for i in range(d):
            printString += ' * (u_{0:d})**{1:.3g}'.format(i+1, A[d*k + i])

        stringList[k] = printString
        print printString

    return stringList

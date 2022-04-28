import os

import random
import numpy as np
import pandas as pd
import plotly.express as px


def random_points(n=2, intervalX=[-1, 1], intervalY=[-1, 1]):
    """
    Generate random list of points in space

    :param n: int
        Number of points
    :param intervalX: list of two itens
        X interval
    :param intervalY: list of two itens
        Y interval
    :return: list
        Random points generated
    """

    list_points = []


    for i in range(n):
        # Generate random point inside defined space
        p = (random.uniform(intervalX[0], intervalX[1]), random.uniform(intervalY[0], intervalY[1]))
        # Ensuring list without duplicates
        while p in list_points:
            p = (random.uniform(intervalX[0], intervalX[1]), random.uniform(intervalY[0], intervalY[1]))

        list_points.append(p)

    return list_points


def make_line(points):

    """
    Generate line using two points
    :param points: list
        Two points used to make a equation
    :return:
    """

    P0 = points[0]
    P1 = points[1]

    a = (P1[1] - P0[1]) / (P1[0] - P0[0])

    if a <= 0:
        sinal = ' +'
    if a > 0:
        sinal = ' -'

    eq = "y"+sinal+ " {}*x".format(abs(a))

    result = P0[1] - (a * P0[0])

    return eq, result


def make_fx(eq, result):

    """
    Make f(x) = ax + b
    :param eq: string
        Equation
    :param result: float
        Equation result
    :return:
    """

    if '+' in eq.split('y')[1].split()[0]:
        sinal1 = ' -'
    else:
        sinal1 = ''

    if result > 0:
        sinal2 = ' +'
    else:
        sinal2 = ' -'

    fx = sinal1 + eq.split('y')[1].split()[1] + sinal2 + " {}".format(abs(result))

    # print("A f(x) utilizada é: {}".format(fx))

    return fx


def evaluate_sample(sample, eq, result):
    """
    Evaluate sample with +1 if > eq or -1 if < eq
    :param sample: list
        List of points to evaluate
    :param eq: string
        Line equation used to evaluate sample
    :param result:
        Result of line equation used to evaluate sample
    :return: list
        Evaluated sample
    """
    evaluated_list = []

    for item in sample:

        x = item[0]
        y = item[1]

        if eval(eq) >= result:
            aux = 1
        if eval(eq) < result:
            aux = -1

        evaluated_list.append([x, y, aux])

    df = pd.DataFrame(evaluated_list, columns=['X', 'Y', 'Value'])

    return df


def plot_eq_sample(fx, sample_evaluate, intervalX=[-1, 1], intervalY=[-1, 1]):
    """
    With a fx and evaluated values plot an graph
    :param fx: string
        Function in format y=ax+b
    :param sample_evaluate: Pandas.DataFrame
        Dataframe with X, Y and Values
    :param intervalX: list of two values
        Limits of space in X axe
    :param intervalY: list of two values
        Limits of space in Y axe
    :return:
    """
    sample_evaluate = sample_evaluate.copy()

    sample_evaluate['Value'] = sample_evaluate['Value'].apply(lambda x: '+1' if x == 1 else '-1')

    fig = px.scatter(sample_evaluate, x='X', y='Y', color='Value')
    x = np.array(intervalX)

    fig = fig.add_trace(px.line(x=x, y=eval(fx))['data'][0])

    fig = fig.update_xaxes(range=intervalX)
    fig = fig.update_yaxes(range=intervalY)

    path_save = os.path.join("/home/wagner/PycharmProjects/IntelComp/PLA/Results", "fxClasses.png")
    fig.write_image(path_save)

def create_sample(n=10):

    # Generate two random points
    two_points = random_points()

    # print("P0 = {} || P1 = {}". format(two_points[0], two_points[1]))

    # Generate line equation
    eq, result = make_line(two_points)

    # Genarate fx
    fx = make_fx(eq, result)

    # Generate sample
    sample = random_points(n)
    # Evaluating sample points
    sample_evaluate = evaluate_sample(sample, eq, result)
    # Plotting sample values and fx
    plot_eq_sample(fx, sample_evaluate)

    return sample_evaluate, eq, fx, result
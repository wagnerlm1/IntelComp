import numpy as np
import pandas as pd
import random
import os

from plotly.subplots import make_subplots
import plotly.express as px

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class PerceptronLinear:

    def __init__(self, training_set, fx='', dimension=2, r=0.2, iterations=1000):

        self.training_set = training_set  # PandasDataframe with Training set (X, Y, Value)
        self.dimension = dimension  # Dimension of Xvector
        self.xVector = training_set[training_set.columns[:dimension]].apply(list, axis=1).to_list()  # Vector with sample
        self.dVector = training_set['Value'].to_list()  # Vector with expected outs
        self.r = r  # Learning rate between 0 and 1
        self.iterations = iterations  # Max number of iterations
        self.fx = fx  # Target function (used only to plot)
        self.g = ''  # String with equation g
        self.weights = []
        self.classification = []
        self.iter_convergence = 0  # Number of iteration to convergence
        self.converged = False  # Checks if PLA converged
        self.classificated = []
        self.iter_weights = {}

    def practice(self):
        """
        Function is used to practice Perceptron Linear Algorithm with a sample
        """

        # Inicialize weights with 0
        for i in range(len(self.xVector[0])):
            self.weights.append(0)

        # Insert w0 = 0
        self.weights.insert(0, 0)

        # Insert x0 = 1
        for item in self.xVector:
            item.insert(0, 1)

        while True:

            for i in range(0, len(self.dVector)):

                xvec = self.xVector[i]  # Each vector in Xvector

                h = self.sign(np.dot(np.array(self.weights), np.array(xvec)))  # h(x) = sign (Sum wi*xi), i e [1,...,d] || h(x) = sign(wT*x)

                # if h(x) = sign(wT*x) different than yn update w
                if h != self.dVector[i]:
                    error = self.dVector[i] - h  # iteration error
                    self.weights = list(np.array(self.weights) + self.r * error * np.array(xvec))  # updating w with learning rate and error

                try:
                    self.classification[i] = h  # try to substitute new classification
                except:
                    self.classification.insert(i, h)  # except insert new classification

            # Saving the weights of each iteration
            self.iter_weights[self.iter_convergence] = self.weights

            # Saving the information of classifications until its finish
            if len(self.classification) == len(self.dVector):
                self.training_set['Iter_{}'.format(self.iter_convergence)] = self.classification

            # Checks if PLA converged
            if self.classification == self.dVector:
                self.converged = True
            else:
                self.converged = False

            # Stopping criterion
            if self.iter_convergence > self.iterations or self.converged:

                # to pring g
                self.g = self.create_g(self.weights)

                break

            # count number of iterations to converge
            self.iter_convergence += 1

    def sign(self, number):
        """
        Verify sign of number
        :param number: float
        :return:
        """
        group = 1 if number >= 0 else -1
        return group

    def create_g(self, w):

        # to print g
        if w[1] >= 0:
            sinal1 = '+'
        else:
            sinal1 = '-'

        if w[2] >= 0:
            sinal2 = '+'
        else:
            sinal2 = '-'

        if w[0] >= 0:
            sinal3 = '-'
        else:
            sinal3 = '+'

        g = '{} {}*x {} {}*y = {} {}'.format(sinal1, abs(w[1]), sinal2, abs(w[2]),
                                                  sinal3, abs(w[0]))
        return g

    def create_gx(self, g):

        # creating gx function
        gx = '(' + g.split()[-2] + ' ' + g.split()[-1] + ' ' + self.invert_sign(g.split()[0]) + \
             ' ' + g.split()[1] + ') / (' + g.split()[2] + ' ' + g.split()[3].split('*')[0] + ')'

        return gx

    def classificate(self, point):
        """
        Fuction classificate a point, after method is trained
        :param point: list with one point
        :return:
        """

        point = list(point[0])

        point.insert(0, 1)

        h = self.sign(np.dot(np.array(self.weights), np.array(point)))
        point.append(h)

        self.classificated = pd.DataFrame([point[1:]], columns=['X', 'Y', 'Value'])

    def invert_sign(self, aux):
        """
        Invert a sign
        :param aux: string
        :return:
        """
        if '+' in aux:
            aux = '-'
        else:
            aux = '+'

        return aux

    def plot(self):

        df = self.training_set.copy()  # dataframe with iterations classifications

        # choosing number of subplots
        n = round(len(df.columns[4:-1])/2)
        n = n if n <= 10 else 10

        # Choosing list of subplots
        to_plot = random.sample(list(df.columns[4:-1]), n)
        to_plot.insert(0, df.columns[3])
        to_plot.insert(len(to_plot), df.columns[-1])
        to_plot = list(set(to_plot))
        to_plot = sorted(to_plot, key=lambda x: int(x.split('_')[1]))

        rows = 0

        if len(to_plot) == 1:
            rows = 1
            cols = 1
        if (len(to_plot) > 1) and (len(to_plot) <= 4):
            rows = 2
            cols = 2
        if (len(to_plot) > 4) and (len(to_plot) <= 6):
            rows = 3
            cols = 2
        if (len(to_plot) > 6) and (len(to_plot) <= 9):
            rows = 3
            cols = 3
        if (len(to_plot) > 9) and (len(to_plot) <= 12):
            rows = 4
            cols = 3

        if rows != 0:

            fig = make_subplots(rows=rows, cols=cols, subplot_titles=tuple(to_plot))

            row = 1
            col = 1

            for item in to_plot:

                iter = int(item.split('_')[1])

                legend = iter == 0

                df[item] = df[item].apply(lambda k: '+1' if k == 1 else '-1')

                scatter_points = px.scatter(df, x='X', y='Y', color=item,
                                            color_discrete_map={'+1': '#00cc00', '-1': '#ff0000'})
                scatter_points = scatter_points.update_traces(showlegend=legend)
                # scatter_points = scatter_points.update_layout(showlegend=legend)

                fig = fig.add_trace(scatter_points['data'][0], row=row, col=col)
                fig = fig.add_trace(scatter_points['data'][1], row=row, col=col)

                #  to plot gx
                x = np.array([-1, 1])

                if self.fx != '':
                    # to plot fx
                    line_fx = px.line(x=x, y=eval(self.fx))
                    line_fx = line_fx.update_traces(name='f(x)', showlegend=legend, line_color='#b36b00')
                    # line_fx = line_fx.update_layout(showlegend=legend)
                    fig = fig.add_trace(line_fx['data'][0], row=row, col=col)

                # to create gx
                w = self.iter_weights[iter]
                g = self.create_g(w)
                gx = self.create_gx(g)

                line_gx = px.line(x=x, y=eval(gx))
                line_gx = line_gx.update_traces(name='g(x)',showlegend=legend, line_color='#3399ff')
                # line_gx = line_gx.update_layout(showlegend=legend)
                fig = fig.add_trace(line_gx['data'][0], row=row, col=col)

                if col == cols:
                    row += 1

                if col != cols:
                    col += 1
                else:
                    col = 1

            fig.update_xaxes(range=[-1, 1])
            fig.update_yaxes(range=[-1, 1])
            fig.update_layout(height=1000, width=1000, legend={'title': 'Legenda'})

            path_save = os.path.join("/home/wagner/PycharmProjects/IntelComp/PLA/Results", "results.png")
            fig.write_image(path_save)





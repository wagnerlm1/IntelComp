import numpy as np
import pandas as pd


class PerceptronLinear:

    def __init__(self, training_set, dimension=2, r=0.2, iterations=1000):

        self.training_set = training_set  # PandasDataframe with Training set (X, Y, Value)
        self.dimension = dimension  # Dimension of Xvector
        self.xVector = training_set[training_set.columns[:dimension]].apply(list, axis=1).to_list()  # Vector with sample
        self.dVector = training_set['Value'].to_list()  # Vector with expected outs
        self.r = r  # Learning rate between 0 and 1
        self.iterations = iterations  # Max number of iterations
        self.g = ''  # String with equation g
        self.weights = []
        self.classification = []
        self.iter_convergence = 0  # Number of iteration to convergence
        self.converged = False  # Checks if PLA converged
        self.classficated = []

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

            for i in range(len(self.dVector)):

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

            # Saving the information of classifications until its finish
            if len(self.classification) == len(self.dVector):
                self.training_set['Iter_'.format(self.iter_convergence)] = self.classification

            # Checks if PLA converged
            if self.classification == self.dVector:
                self.converged = True
            else:
                self.converged = False

            # Stopping criterion
            if self.iter_convergence > self.iterations or self.converged:

            #---------------------------------------------------------------------------------------------------------X
                # to print g
                if self.weights[1] >= 0:
                    sinal1 = '+'
                else:
                    sinal1 = '-'

                if self.weights[2] >= 0:
                    sinal2 = '+'
                else:
                    sinal2 = '-'

                if self.weights[0] >= 0:
                    sinal3 = '+'
                else:
                    sinal3 = '-'

                self.g = '{} {}*x {} {}*y = {} {}'.format(sinal1, abs(self.weights[1]), sinal2, abs(self.weights[2]),
                                                          sinal3, abs(self.weights[0]))
            # --------------------------------------------------------------------------------------------------------X

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

        self.classficated = pd.DataFrame([point[1:]], columns=['X', 'Y', 'Value'])



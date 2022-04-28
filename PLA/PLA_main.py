import pandas as pd

from PLA_model import PerceptronLinear
from create_sample import create_sample, random_points, evaluate_sample

def main():

    n_runs = 1000  # Number o runs
    n = 100   # Number of itens in samples

    listIterConvergence = []
    list_of_probabilities = []

    for i in range(n_runs):

        if i % 50 == 0:
            print('Running on {} lap..'.format(i))
        # Create random samples

        sample_evaluate, eq, fx, result = create_sample(n)
        # Test to practice learning
        test = PerceptronLinear(sample_evaluate)
        test.practice()

        listIterConvergence.append(test.iter_convergence)

        # To get probability of F and g disagree in their classification of a random point
        aux_P = []
        for v in range(100):

            point = random_points(1)  # Generate one random point
            test.classificate(point)  # Classificating point in g

            f_classification = evaluate_sample(point, eq, result)  # Classificating point in f
            g_classification = test.classficated  # Save g classification

            # If classifications are the same add 1 to aux_P else add 0
            if f_classification['Value'].values[0] == g_classification['Value'].values[0]:
                aux_P.append(1)
            else:
                aux_P.append(0)

        F_diff_G = 1-(pd.Series(aux_P).mean())  # Probability of F be different from G
        list_of_probabilities.append(F_diff_G)  # List of Probabilities


    print('A média de iterações, em {}, runs foi: {}'.format(n_runs, pd.Series(listIterConvergence).mean()))
    print('A média da probabilidade de f e g discordarem foi: {}'.format(pd.Series(list_of_probabilities).mean()))



if __name__== "__main__":
    main()




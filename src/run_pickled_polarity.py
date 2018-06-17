from src.data.import_dataset import   read_pickle
from src.features.sentiment import  run_model
def main():
    matrix_training = read_pickle('', 'training_matrix_hui')
    matrix_testing = read_pickle('', 'testing_matrix_hui')

    model = run_model(matrix_training, matrix_testing)



if __name__ == '__main__':
    main()

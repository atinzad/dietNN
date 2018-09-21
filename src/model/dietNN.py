
from argparse import ArgumentParser
from keras.models import load_model
from keras.models import model_from_json



def build_parser():
    par = ArgumentParser()
    par.add_argument('--m', type=str,
                     dest='model_load_path', help='filepath to load model in json format', required=True)
    par.add_argument('--w', type=str,
                     dest='weights_load_path', help='filepath to load weights in h5 format', required=True)
    par.add_argument('--c', type=int,
                    dest='precent_of_prunning', help='percent of parameters to be prunned', required=True)

    return par




if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()
    model_load_path = options.model_load_path
    weights_load_path = options.weights_load_path
    precent_of_prunning = options.precent_of_prunning
    
    print (model_load_path)
    print (weights_load_path)
    
    # load json and create model
    with open(model_load_path, 'r')as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_load_path)
    print("Loaded model from disk")
    
    print (loaded_model.summary())
    
    
    number_of_params_to_prune = precent_of_prunning/100* loaded_model.count_params()
    
    
    for i in range(number_of_params_to_prune):
        pass
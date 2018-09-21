
from argparse import ArgumentParser
from keras.models import load_model
from keras.models import model_from_json
from kerassurgeon.operations import delete_channels
from keras.utils.vis_utils import plot_model
import copy
import random



def build_parser():
    par = ArgumentParser()
    par.add_argument('--m', type=str,
                     dest='model_load_path', help='filepath to load model in json format', required=True, 
                     default="/home/atinzad/dietNN/data/raw/model.json")
    par.add_argument('--w', type=str,
                     dest='weights_load_path', help='filepath to load weights in h5 format', required=True,
                     default="/home/atinzad/dietNN/data/raw/model.h5")
    par.add_argument('--c', type=int,
                    dest='precent_of_prunning', help='percent of parameters to be prunned', required=True, default = 30)

    return par


#python dietNN.py --m /home/atinzad/dietNN/data/raw/model.json --w /home/atinzad/dietNN/data/raw/model.h5 --c 30

if __name__ == "__main__":
#    parser = build_parser()
#    options = parser.parse_args()
#    model_load_path = options.model_load_path
#    weights_load_path = options.weights_load_path
#    precent_of_prunning = options.precent_of_prunning
    
    model_load_path = "/home/atinzad/dietNN/data/raw/model.json"
    weights_load_path = "/home/atinzad/dietNN/data/raw/model.h5"
    precent_of_prunning = 1
    
    print (model_load_path)
    print (weights_load_path)
    
    # load json and create model
    with open(model_load_path, 'r')as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_load_path)
    print("Loaded model from disk")
    
    print (loaded_model.summary())
    
    
    number_of_params_to_prune = int(precent_of_prunning/100* loaded_model.count_params())
    
    final_size =  loaded_model.count_params() - number_of_params_to_prune
    
    saved_model = copy.copy(loaded_model)
    
    trainable_layers = [layer for layer in saved_model.layers if len(layer.trainable_weights)>0]
    
    trainable_layers_max=[0]*len(trainable_layers)
    
    for i, layer in enumerate(trainable_layers):
        if 'units' in layer.get_config():
            trainable_layers_max[i]=(layer,layer.get_config()['units'])
        elif 'filters' in layer.get_config():
            trainable_layers_max[i]=(layer,layer.get_config()['filters'])
            
        
    while saved_model.count_params() > final_size:
        
        current_no = saved_model.count_params()
        #randomly choose a layer
        layer, maxn =random.choice(trainable_layers_max)
        
        #randomly choose a parameter
        i = random.randint(0,maxn-1)
        
        try:
            saved_model = delete_channels(saved_model, layer, [i])
        except:
            pass
        
        print (saved_model.count_params(),layer.name, i)
        
        if current_no == saved_model.count_params():
            trainable_layers = [layer for layer in saved_model.layers if len(layer.trainable_weights)>0]
    
            trainable_layers_max=[0]*len(trainable_layers)
    
            for i, layer in enumerate(trainable_layers):
                if 'units' in layer.get_config():
                    trainable_layers_max[i]=(layer,layer.get_config()['units'])
                elif 'filters' in layer.get_config():
                    trainable_layers_max[i]=(layer,layer.get_config()['filters'])


    model_json = saved_model.to_json()
    with open("model_small.json", "w") as json_file:
        json_file.write(model_json)
    saved_model.save_weights("model_small.h5")
    print("Saved model to disk")
    try:
        plot_model(saved_model, to_file='model_small.png')
    except:
        pass
        
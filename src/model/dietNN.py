
from argparse import ArgumentParser
from keras.models import load_model
from keras.models import model_from_json
from kerassurgeon.operations import delete_channels
from keras.utils.vis_utils import plot_model
from keras import layers
import copy
import random



def build_parser():
    #par = ArgumentParser()
    par = ArgumentParser(fromfile_prefix_chars='@')
    par.add_argument('--m', type=str,
                     dest='model_load_path', help='filepath to load model in json format', required=True, 
                     default="/home/atinzad/dietNN/data/raw/model.json")
    par.add_argument('--w', type=str,
                     dest='weights_load_path', help='filepath to load weights in h5 format', required=True,
                     default="/home/atinzad/dietNN/data/raw/model.h5")
    par.add_argument('--c', type=int,
                    dest='precent_of_prunning', help='percent of parameters to be prunned', required=True, default = 30)
    par.add_argument('--q', type=bool,
                    dest='invoke quantization', help='True if quantization', required=True, default = False)

    return par


#python dietNN.py --m /home/atinzad/dietNN/data/raw/model.json --w /home/atinzad/dietNN/data/raw/model.h5 --c 30
    

def prune(model, layer, rand=True,no_of_weights=0, weights=[]):
    try:
        if 'units' in layer.get_config():
            maxn=layer.get_config()['units']
        elif 'filters' in layer.get_config():
            maxn=layer.get_config()['filters']
    except:
        print ("layer is not a dense or conv layer")
        return model
        
    if rand:
        if no_of_weights==0:
            print ("did not specifiy no_of_weights")
            return model
        elif no_of_weights >= maxn:
            print ("number_of_weights greater than or equal to what is supported by layer")
            return model
        else:
            data = list(range(maxn))
            random.shuffle(data)
            weights = [data[i] for i in range(no_of_weights)]
            
    else:
        if len(weights)==0:
            print ("did not specifiy weights")
            return model
        elif max(weights) > maxn:
            print ("at least one of the weights is out of bounds")
            return model

    model = delete_channels(model, layer, weights)
    return model
                
            
def quantize(model, layer) :
    weights = layer.get_weights()
    new_weights=copy.copy(weights)
    for i,w in enumerate(weights):
        new_weights[i]= w.astype('float16')
        new_weights[i] = new_weights[i]+1
    for i,l in enumerate(model.layers):
        if l.name == layer.name:
            my_i = i
            break
    
    config = layer.get_config()
    new_layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
    
    new_layer.set_weights(new_weights)
    
    model = replace_intermediate_layer_in_keras(model, my_i, new_layer)
    
    
    #model.layers[my_i].set_weights(new_weights)
    #model.get_layer(name=layer.name).set_weights(new_weights)
    return model
    
    
def replace_intermediate_layer_in_keras(model, layer_id, new_layer):
    from keras.models import Model

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model    
    
    
        
        
    

if __name__ == "__main__":
    
    parser = build_parser()
    options = parser.parse_args()
    model_load_path = options.model_load_path
    weights_load_path = options.weights_load_path
    precent_of_prunning = options.precent_of_prunning
    
#    model_load_path = "/home/atinzad/dietNN/data/raw/model.json"
#    weights_load_path = "/home/atinzad/dietNN/data/raw/model.h5"
#    precent_of_prunning = 30
#    quantization = True
    
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
    
    print (saved_model.summary())
    
    while saved_model.count_params() > final_size:
    
        trainable_layers = [layer for layer in saved_model.layers if len(layer.trainable_weights)>0]
        
        random.shuffle(trainable_layers) 
        
        for layer in trainable_layers:
            
            try:
                if 'units' in layer.get_config():
                    maxn=layer.get_config()['units']
                elif 'filters' in layer.get_config():
                    maxn=layer.get_config()['filters']
            except:
                print ("layer is not a dense or conv layer")
                continue
            
            
            try:
                saved_model = prune(saved_model, layer, rand=True, no_of_weights=int(precent_of_prunning/100*maxn))
                if quantization:
                    saved_model = quantize(saved_model, layer)
            except:
                print ("could not process", layer.get_config()['name'])
            
            if saved_model.count_params() <= final_size:
                break
    
    
    
    
    print (saved_model.summary())
    
    model_json = saved_model.to_json()
    with open("model_small.json", "w") as json_file:
        json_file.write(model_json)
    saved_model.save_weights("model_small.h5")
    print("Saved model to disk")
    try:
        plot_model(saved_model, to_file='model_small.png')
    except:
        pass
        
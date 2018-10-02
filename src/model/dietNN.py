
from argparse import ArgumentParser
from keras.models import model_from_json
from kerassurgeon.operations import delete_channels
from keras.utils.vis_utils import plot_model
from keras import layers
import copy
import random
import os
from keras.preprocessing import image
from keras import optimizers 
import numpy as np


def build_parser():
    #par = ArgumentParser()
    par = ArgumentParser(fromfile_prefix_chars='@')
    par.add_argument('--m', type=str,
                     dest='model_load_path', help='filepath to load model in json format', required=True)
    par.add_argument('--w', type=str,
                     dest='weights_load_path', help='filepath to load weights in h5 format', required=True)
    par.add_argument('--d', type=str,
                     dest='dataset_load_path', help='filepath to load validation dataset', required=True)
    par.add_argument('--c', type=int,
                    dest='precent_of_prunning', help='percent of parameters to be prunned', required=True)
    par.add_argument('--q', type=bool,
                    dest='invoke_quantization', help='True if quantization', required=True, default = False)

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


def prune_genetic(model, layer, validation_generator, no_of_weights=1, loops=1):
    try:
        if 'units' in layer.get_config():
            maxn=layer.get_config()['units']
        elif 'filters' in layer.get_config():
            maxn=layer.get_config()['filters']
    except:
        print ("layer is not a dense or conv layer")
        return model
    
    results = []
    weights_configs=[]
    models=[]
    data = list(range(maxn))
    for i in range(loops):
        random.shuffle(data)
        weights = [data[i] for i in range(no_of_weights)]
        try:
            model_try = delete_channels(model, layer, weights)
        except:
            print ("broke at",i)
            break
        
        model_try.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])
        results.append(model_try.evaluate_generator(generator=validation_generator, steps=10)[1])
        weights_configs.append(weights)
        models.append(copy.copy(model_try))
    
   
    try:
        best_i = np.argmax(results)
        model = delete_channels(model, layer, weights_configs[best_i])
        print (results)
    except:
        print ("No update occurend for layer", layer.name)
        pass
        
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
    
#    parser = build_parser()
#    options = parser.parse_args()
#    model_load_path = options.model_load_path
#    weights_load_path = options.weights_load_path
#    dataset_load_path = options.dataset_load_path
#    precent_of_prunning = options.precent_of_prunning
#    quantization =False
    
    

    
    model_load_path = '~/dietNN/src/preprocess/catdog_model.json'
    weights_load_path = '~/dietNN/src/preprocess/catdog_model.h5'
    dataset_load_path = '~/dietNN/data/raw/catdog/test'
    precent_of_prunning = 30
    quantization =False
    
    
    if model_load_path[0]=="~":
        model_load_path = os.path.expanduser(model_load_path)
        
    if weights_load_path[0]=="~":
        weights_load_path = os.path.expanduser(weights_load_path)
    
    if dataset_load_path[0]=="~":
        dataset_load_path = os.path.expanduser(dataset_load_path)
    
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
    
    for layer in saved_model.layers:
        layer.trainable = True
    
    print (saved_model.summary())
    img_size=224
    batch_size=10
    
    saved_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])
    
    test_datagen = image.ImageDataGenerator(rescale=1. / 255)
    
    validation_generator = test_datagen.flow_from_directory(
                    dataset_load_path,
                    target_size=(img_size,img_size),
                    batch_size=batch_size,
                    class_mode='categorical')
    
    while saved_model.count_params() > final_size:
    
        trainable_layers = [layer for layer in saved_model.layers if len(layer.trainable_weights)>0]
        
        prediction_layer = list(saved_model.layers)[-1]
        
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
            
            if layer.name != prediction_layer.name:
                try:
                    saved_model = prune_genetic(saved_model, layer,  validation_generator, no_of_weights=int(precent_of_prunning/100*maxn), loops=20)
                    print (saved_model.count_params())
                    #saved_model = prune(saved_model, layer, rand=True, no_of_weights=int(precent_of_prunning/100*maxn))
                    #if quantization:
                    #    saved_model = quantize(saved_model, layer)
                except:
                    print ("could not process", layer.get_config()['name'])
            
            
            
            
            
#            saved_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])
#            saved_model.evaluate_generator(generator=validation_generator)
            
            if saved_model.count_params() <= final_size:
                break
    
    
    
    import time
    print (saved_model.summary())
    saved_model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])
    start = time.time()
    print(saved_model.evaluate_generator(generator=validation_generator, steps=20))
    print(time.time()-start)
    
    model_json = saved_model.to_json()
    with open("model_small.json", "w") as json_file:
        json_file.write(model_json)
    saved_model.save_weights("model_small.h5")
    print("Saved model to disk")
    try:
        plot_model(saved_model, to_file='model_small.png')
    except:
        pass
        
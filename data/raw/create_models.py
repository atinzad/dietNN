#Create Model and save in h5 and json files

from keras.applications.vgg16 import VGG16 
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
import os
from pathlib import Path




model = VGG16()
print(model.summary())
# serialize model to JSON
model_json = model.to_json()

    
my_file = Path(os.getcwd() + "/model.json")
if not my_file.exists():
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")
    try:
        plot_model(model, to_file='vgg.png')
    except:
        pass



#Test to laod checkpoint


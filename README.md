# DietNN
Tool to reduce the size of Neural Networks

ToDo:
- Train Models in Keras (get accuracy)
- Run Model (inference)
- Add data (imagenet) to the pipeline
- Run data in dietNN to check accuracy
- Have examples in tests
- Refactor dienNN.py

## Motivation for this project format:
- **src** : Put all source code for production within structured directory
- **tests** : Put all source code for testing in an easy to find location
- **configs** : Enable modification of all preset variables within single directory (consisting of one or many config files for separate tasks)
- **data** : Include example a small amount of data in the Github repository so tests can be run to validate installation
- **build** : Include scripts that automate building of a standalone environment
- **static** : Any images or content to include in the README or web framework if part of the pipeline

## Setup
Clone repository and update python path
```
repo_name=dietNN # URL of your new repository
username=atinzad # Username for your personal github account
#Clone master
git clone https://github.com/$username/$repo_name
#Or clone a particular branch
my_branch=setup_20180918
git clone -b $my_branch https://github.com/$username/$repo_name

cd $repo_name
echo "export $repo_name=${PWD}" >> ~/.bash_profile
echo "export PYTHONPATH=$repo_name/src:${PYTHONPATH}" >> ~/.bash_profile
source ~/.bash_profile
```
Create new development branch and switch onto it
```
branch_name=dev-readme_requisites-20180917 # Name of development branch, of the form 'dev-feature_name-date_of_creation'}}
git checkout -b $branch_name
git push origin $branch_name
```

## Requisites
- Python 3.6.5
- Tensorflow 1.10.1 (pip install tensorflow #for latest version)
- Numpy 1.14.3 (pip install numpy #for latest version)
- Keras 2.2.2 (pip install keras #for latest version)
- Kerassurgeon 0.1.1 (pip install kerassurgeon #for latest version)
- **Optional**: GraphViz (sudo apt-get install graphviz)
- **Optional**: Pydot 1.2.4 (pip install pydot #for latest version)


## To install Requisits
```
cd $repo_name
pip install -r requirements.txt
```
#Once done make sure Tensorflow as running as backend (most likely it is)
#In python, import keras then go back to shell (this will create keras.json config file)
```
python
import keras
exit()
```

#Edit $HOME/.keras/keras.json
```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

```

## Fetch and create h5 parameter file and json model file
```
cd ~/dietNN/data/raw
python create_models.py #this will create model.json (in KB range) and model.h5 (in MB range)
```

## Run dietNN.py
#example on model.json and model.h5 with reduction request of ~30% in footprint
```
cd ~/dietNN/src/model
python dietNN.py --m ~/dietNN/data/raw/model.json --w ~/dietNN/data/raw/model.h5 --c 30
```

#Alternativly, using a myconfig.txt file
#contents of myconfig.txt file
```
--m=~/dietNN/data/raw/model.json
--w=~/dietNN/data/raw/model.h5
--c=30
```

#In the command prompt
```
python dietNN.py @myconfig.txt
```
# model_small.json and model_small.h5 will be produced and stored in ~/dietNN/src/model folder
#Note that model_small.h5 is ~30% smaller than model.h5


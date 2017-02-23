# keras


##Confguration 

If you tke a look at the configuration/config file. It is divided into three four parts Framework, layers, Connections and Models.
  
###LAYERS:  
* ```convolayer1	convo2d	{64, 3, 3, activation='relu', border_mode='same', dim_ordering='th'}```
The above config line represents  
```<layer name> <layer primary key> <layer properties>```  
  
* Each line will have the description of the layer.
  
* If a layer has same properties as another layer defined before, then we can write  
```<layer name> @<layer name (of the layer from which we will reuse those properties)>```  
  
* We can define a model in this configuration (covered later here). Once defined, we can reuse the model (with its weights) as a primitive
layer like this
```<layer name> @@<model name>```
  
  
###CONNECTIONS:   
* ```textinput1 > embedlayer1 > lstmlayer1```
represents that the layer named 'textinput1' inputs to the layer named 'embedlayer1'.  
  
* The same things can also be put into the config as  
```textinput1 > embedlayer1  
embedlayer1 > lstmlayer1```  
  
* ```videoinput1 + useimagemod : timedistributelayer1```  
represents that videoinput1 and useimagemod merge into the layer timedistributelayer1. We can merge any number of layers with this general
format  
```layer1 + layer2 + ... + layern = mergedlayer```  

###MODELS:  
* ```imageinput1 , textinput1 -> denselayer1 : textandimagemod```  
here we define a model named  textandimagemod with imageinput1 and textinput1 as inputs and denselayer1 as the output  

##Usage  

* Create a configuration file  
* Run ```python generate.py <configFilePath>```
* Creates a file gencode/auto.py which is the generated code
* Creates a jupyter notebook notebooks/notebook.ipynb




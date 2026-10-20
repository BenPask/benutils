[![npm version](https://img.shields.io/npm/v/@benp2325/benutils?color=red)](https://www.npmjs.com/package/@benp2325/benutils)
![JavaScript](https://img.shields.io/badge/Vanilla_JS-F7DF1E?logo=javascript&logoColor=black)
![Dependencies](https://img.shields.io/badge/dependencies-zero-success)

# @benp2325/benutils

A zero-dependency, vanilla JavaScript library built completely from scratch for low-level matrix manipulation and neural network modeling. Also includes a custom scatterplot creation module that returns an HTML canvas element. 

Public npm package page:
https://www.npmjs.com/package/@benp2325/benutils?activeTab=readme

This project was created as a learning exercise to develop first-principle neural network engineering, troubleshooting, and analysis experience. Additionally, it was created alongside and served as an engine for an interactive model creation/visualization web-application that I led a student team to create. The application is being hosted online and doubles as a presentation/evaluation tool for this package. Extensive features are available and all run fully in the browser. Created models are trained on real-world data scraped from an API. [Link to VFL NN Model Studio](https://github.com/BenPask/VFL-NN-Model-Studio). 

Jump to [Model Functions](#model-functions)

# Getting Started

You can dowload benutils and its sub-package, vis, simply via npm:

```
npm i @benp2325/benutils
```

You can also import benutils into your web projects by pasting:
```
<script type="importmap">
{
  "imports": {
    "@benp2325/benutils": "https://cdn.jsdelivr.net/npm/@benp2325/benutils@1.1.0/index.js",
    "@benp2325/benutils/vis": "https://cdn.jsdelivr.net/npm/@benp2325/benutils@1.1.0/src/vis.js"
  }
}
</script>
```
Into the header of your projects html file. 

# Creating A Model

Creating and training a neural network model using benutils only requires a few simple method calls.

 you can create a model via:
```
const model = benutils.neuralnet()
```

Creating a model this way will result in the default values for every argument. 

### Arguments for benutils.neuralnet()

The .neuralnet() function accepts a configuration object to customize various properties of the model.

```
benutils.neuralnet({

  hidden_layer_info: {
    count:<integer>, //Number of hidden layers

    init_size:5, //Neuron count of the first hidden layer, other layer sizes are determined by architecture 

    style:<string>, //"Flat", "Po2", or "Hourglass". For a detailed look at how architectures are 
    applied, see the architecture types section of this document. 

    hourglass_operation:<string> //"/", "*", "**" or "sqroot". Only applies to models with the 
    "Hourglass" architecture type. 
  },

  activation_type: <string> //"relu", "leaky_relu", "tanh", "sigmoid"

  learn_rate: <float>, //Constant that delta values will be multiplied by to smooth learning

  output_layer_info: {
    init_size: <integer>, //For exclusively regression based implementation, should remain 1

    activation: <string> //Activation for the output layer, same options as activation_type
  }, 

  epochs: <integer>, //Number of times the model runs through every observation

  batch_size: <integer>, //Interval, number of forward passes before applying delta graidents

  tt_split: <float>, //Value from 0-1, represents the percentage of the data reserved for testing

  random_seed: <integer>, //Number that determines the sequence of random generated values, 

  metric_report_interval: <integer> //Interval, number of obervations between the calculation and 
  storage of performance metrics that require denormalization
})
```

### Importing/setting features
Firstly, make sure to store your input features as either a JavaScript object with string labels as the keys and arrays as the values or a .csv file stored as a variable.

_Regardless of importing method, all features must have an equal number of observations and their target feature must be labeled TARGET in all caps._

As an object:
```
const input_data = {
    "FEAT1":[42, 346, 256, 2324, 5234, 42, 23, 78],
    "FEAT2":[352, 234, 23, 234, 23, 46, 77, 32],
    "FEAT3":[3222, 234, 56, 234, 78, 46, 97, 1],
    //TARGET FEATURE MUST BE LABELED "TARGET" EXACTLY AS SHOWN
    "TARGET":[22, 44, 34, 346, 3456, 342, 23, 277]
}
```

For the .set_features_csv() method, a JS object that specifies the selected columns via index must be passed. This is for ease, as it allows users to import whole datasets and select/switch features as they work. The method automatically detects and strips headers, so there is no need to remove them. 

As a .csv (stored as a variable):
```
import {rawCsvData} from '/path-to-file.csv'

feature_selection = {
    "FEAT1":0,
    "FEAT2":5,
    "FEAT3":3,
//TARGET FEATURE MUST BE LABELED "TARGET" EXACTLY AS SHOWN
    "TARGET":6
}
```

Data will not be passed into the model until you call model.train() later in the program.

Once your input data is stored in one of those 2 compatible formats,

### Normalizing Feature Data
Normalizing your feature data in benutils (if not already normalized) is a single method call made before training and testing.

Feature maximum and minimums are stored during the setting methods, the only method that must be called to normalize all features is:

```
benutils.normalize_features(model)
```

### Training and Testing Models

To run your model through its training procedures and store it's performance metrics, call:
```
model.train();
```

To run your model through its testing set and store it's testing performance, call:
```
model.test();
```

### Predicting Single Observations

To use a trained model to predict a single observation, call:
```
model.predict({
  "FEAT1":<numerical value>
  "FEAT2":<numerical value>
  "FEAT3":<numerical value>
  ...
})  
```
The method returns an array with the predicted value as the first element and the layer values that led to it. 0 (be aware that model.properties.layers can easily become a large object to return for many consecutive predictions on larger models)

### Model Evaluation & Performance Metrics

Every training and testing method call stores train/test performance metrics within the model's data. Specifically, performance metrics are stored in:
```
model.properties.performance_metrics.training //.loss_list .mse_list .rmse_list .mae_list .r2
OR
model.properties.performance_metrics.testing //.loss .mse .rmse .mae .r2
```
_to differentiate them in reporting, loss refers to normalized error, while mse refers to the denormalized error_








[![npm version](https://img.shields.io/npm/v/@benp2325/benutils?color=red)](https://www.npmjs.com/package/@benp2325/benutils)
![JavaScript](https://img.shields.io/badge/Vanilla_JS-F7DF1E?logo=javascript&logoColor=black)
![Dependencies](https://img.shields.io/badge/dependencies-zero-success)

# @benp2325/benutils

**benutils is a zero-dependency JavaScript library for matrix manipulation and neural-network modeling, implemented from scratch. It includes configurable network architectures, training and evaluation utilities, and a custom Canvas-based scatterplot module. It serves as the engine for VFL NN Model Studio, an interactive application for training and inspecting models in the browser.**

The project, originating in a passionate drive to understand and recreate the complex nature of neural networks, resulted in learning opportunities, skill development, and a published, fully reusable API.

Public npm package page:
https://www.npmjs.com/package/@benp2325/benutils?activeTab=readme

[Link to VFL NN Model Studio](https://github.com/BenPask/VFL-NN-Model-Studio). 
<img width="940" height="540" alt="image" src="https://github.com/user-attachments/assets/3c6e065d-f8ea-4f22-91c5-3b779616d80c" />

Jump to: [Model Demonstration](#model-demonstration) [Creating A Model](#creating-a-model)

# Getting Started

You can download benutils and its sub-package, **vis**, via npm:

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
Into the header of your project's HTML file. 

# Model Demonstration

The following is a start-to-finish reproducible example of creating and training a model with benutils. 

In this example, the model is trained on the [Boston Housing Dataset](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset) 

_housing.js, as it appears in this demonstration, is available in the [test folder](https://github.com/BenPask/benutils/tree/main/tests)._

Target: MEDV (Median value of home); 13 input features, 506 observations.

```
import * as benutils from "@benp2325/benutils";
import {rawCsvData} from './housing.js'

const feature_selection = {
    "TARGET":13,
    //put features below
    "CRIM":0,
    "ZN":1,
    "INDUS":2,
    "CHAS":3,
    "NOX":4,
    "RM":5,
    "AGE":6,
    "DIS":7,
    "RAD":8,
    "TAX":9,
    "PTRATIO":10,
    "B":11,
    "LSTAT":12
}

const model = benutils.neuralnet({
    hidden_layer_info: {
        count: 2, //Model will have 2 hidden layers
        init_size: 64, 
        style: "po2" //First hidden layer has 64 neurons as indicated above, po2 architecture means second(only other) will have 32
    },
    activation_type: "relu",
    learn_rate: .01,
    output_layer_info:{
        init_size: 1, 
        activation: "none"
    },
    epochs: 100,
    tt_split: 0.25,
    random_seed: 58373 // specified seed makes training reproducible; VFL NN STUDIO randomizes it by setting seeds randomly based on user interaction. 
});

model.set_features_csv(rawCsvData, feature_selection)
benutils.normalize_features(model)

model.train();
model.test();
```

Here is an extensive look at how the model performed. For both sets, "loss" represents normalized error, while "mse" represents unnormalized error.
```
console.log
      {
        training: {
          loss_list: [
             0.17850832768640126,   0.0685272905394938,  0.04586903015957187,
            0.037040204136371456,  0.03260680466347627,  0.02978100137337426,
            0.027661733755270448, 0.025973928543709363,  0.02456985654707905,
            0.023392421999818596,  0.02236151447870269, 0.021441618108344343,
            0.020605727445190146, 0.019858315860311223, 0.019185146020100456,
             0.01858292700413709, 0.018039212366171342, 0.017548423634609872,
             0.01709818686596367, 0.016678923125927444, 0.016291911395665968,
             0.01591924956833477, 0.015576374152313381, 0.015253100129289142,
             0.01495117193381494, 0.014664430332045295, 0.014395996457491659,
            0.014145963162003448, 0.013903047970257683, 0.013681379907369772,
            0.013465944705328296, 0.013265225500936404, 0.013075912049262473,
            0.012900715683011143, 0.012731724650699506, 0.012569578942039594,
            0.012415116384417188,  0.01226823703418272, 0.012130751864648362,
            0.011992060481679606, 0.011861522329414504, 0.011737431766049014,
              0.0116171144809487, 0.011501615315932094,  0.01139276308755255,
            0.011282811348963586, 0.011179872897957265, 0.011080027967481676,
            0.010982374346109708, 0.010888981428774345, 0.010802966387201313,
             0.01071214187809332, 0.010626747396956812,  0.01054694828975689,
            0.010464781511764873, 0.010388153784053961, 0.010315031895721943,
            0.010239035408545453,  0.01016725749551249, 0.010094504702733473,
            0.010027265312008161, 0.009955723335510761, 0.009888124491509945,
            0.009825783158407765, 0.009756207428963522,  0.00969197883196062,
             0.00962875544676616, 0.009570151204737984, 0.009506018024391685,
            0.009449493191551873, 0.009390372809894233,  0.00933062085615058,
             0.00927576175949905, 0.009219033044976857, 0.009160642324355824,
            0.009107865785952653, 0.009052725333149016, 0.008996118504491026,
             0.00894631607231904,  0.00889423674638184, 0.008843203000831112,
            0.008787870649045933, 0.008739760017269433,  0.00869054993818197,
            0.008637809093083294, 0.008583940300474362,  0.00853784517023919,
            0.008490438081561419, 0.008445101884932523, 0.008397394497448816,
            0.008354680361306313, 0.008311433620691661, 0.008267172126406472,
            0.008223094206303603, 0.008185460051715263, 0.008143824278573127,
            0.008100077098557583,   0.0080657212411584, 0.008025761025322013,
            0.007987963273066345
          ],
          mse_list: [
            361.47936356496234, 138.76776334247498,    92.884786073133,
             75.00641337615217,  66.02877944353938,   60.3065277810829,
              56.0150108544227,  52.59720530101144, 49.753959507835056,
            47.369654549632635,  45.28206681937298,  43.41927666939727,
             41.72659807651001,   40.2130896171302,   38.8499206907034,
            37.630427183377584,  36.52940504149697,  35.53555786008503,
             34.62382840357642,  33.77481933000305, 32.991120576223565,
              32.2364803758779, 31.542157658434604, 30.887527761810524,
            30.276123165975317, 29.695471422391705,  29.15189282642066,
            28.645575403056956,  28.15367213977178, 27.704794312423807,
            27.268538028289814,  26.86208163939626, 26.478721899756486,
              26.1239492580976, 25.781742417666504, 25.453397357630177,
            25.140610678444787, 24.843179994220034, 24.564772525912932,
            24.283922475401212,  24.01958271706436, 23.768299326249245,
            23.524656823921074, 23.290771014762484, 23.070345252293922,
            22.847692981651285,  22.63924261836345, 22.437056634150412,
             22.23930805087217, 22.050187393268043, 21.876006934082657,
            21.692087303138965, 21.519163478837534,   21.3575702867577,
            21.191182561323888, 21.036011412709296, 20.887939588836918,
            20.734046702304557, 20.588696428412803,  20.44137202303528,
            20.305212256816514,  20.16033975440929, 20.023452095307647,
            19.897210895775732, 19.756320043651126, 19.626257134720284,
            19.498229779701493, 19.379556189594407, 19.249686499393178,
            19.135223712892525,  19.01550494003582,  18.89450723370492,
            18.783417562985576, 18.668541916078123, 18.550300706820543,
             18.44342821655411, 18.331768799626754, 18.217139971594314,
             18.11629004644607,  18.01082941142322, 17.907486076682982,
            17.795438064318017,   17.6980140349706,  17.59836362481848,
             17.49156341349367, 17.382479108460576,  17.28913646973435,
            17.193137115161893, 17.101331316988343,  17.00472385733386,
             16.91822773164525,  16.83065308190061,   16.7410235559731,
            16.651765767764804, 16.575556604723417,  16.49124416411057,
            16.402656124579114, 16.333085513345758, 16.252166076277096,
             16.17562562795933
          ],
          rmse_list: [
            19.012610645699404,
            6.729195109325705,
            5.7437897399037485,
            5.221928573648802,
            4.900977730725202,
            4.677179378010069,
            4.506130519283314,
            4.360677119443244,
            4.231723771311518,
            4.1131773280087565
          ],
          mae_list: [
            15.261668949079331,
            4.826682199995472,
            4.2022149842548755,
            3.8495307945813964,
            3.622968784514398,
            3.450928130254514,
            3.3127638733353235,
            3.1903637554423643,
            3.087837083139308,
            2.994473355702672
          ],
          r2_list: [
            -3.1199960055354747,
            0.4838932641180237,
            0.6239805126033061,
            0.6892042006343854,
            0.7262344830062488,
            0.7506660953017598,
            0.7685693795502112,
            0.783268943423078,
            0.7958976955765911,
            0.8071728632370696
          ]
        },
        testing: {
          loss: 0.007787764694161667,
          mse: 5.350449411237275,
          rmse: 3.9711740714400037,
          r2: 0.785846264199548,
          mae: 3.027697920962045
        }
      }
```

Training the same model through [VFL NN Model Studio](https://github.com/BenPask/VFL-NN-Model-Studio), allows us to visualize the model, verify consistency/reproducability of our results, and get additional evaluation capabilities from scatter plots.

<img width="1834" height="919" alt="image" src="https://github.com/user-attachments/assets/1f83c2d2-3959-478c-aa91-f9aa74b3301d" />
<img width="1274" height="716" alt="image" src="https://github.com/user-attachments/assets/ffcd757b-3453-4ead-854f-681686fce596" />
<img width="842" height="632" alt="image" src="https://github.com/user-attachments/assets/461c9729-4b45-4325-ba5c-13b5d02d6da2" />
<img width="842" height="632" alt="image" src="https://github.com/user-attachments/assets/dfd36fa3-1415-4c2f-8dd4-591cb727e11c" />

On this train/test split, the model achieved a test-set R2 of approximately 0.786, an RMSE of 3.97, and an MAE of 3.03. The visualization provides a view of the prediction errors and training behavior.

The scatter plots show us that many values for the target variable, MEDV, are placed exactly at 50. This is due to censoring and has been established as an issue with the [Boston Housing Dataset](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset), as evidenced by [this](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) report from the University of Toronto in 1996. The dataset was chosen as a legacy demonstration to show model capabilities, rather than a benchmark for real-world housing predictions.

**For this dataset, we have verified the capability of the model and reproduced results via the seed.**

# Creating A Model

Creating and training a neural network model using benutils only requires a few simple method calls.

You can create a model via:
```
const model = benutils.neuralnet();
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

  activation_type: <string>, //"relu", "leaky_relu", "tanh", "sigmoid"

  learn_rate: <float> //Constant that delta values will be multiplied by to smooth learning

  output_layer_info: {
    init_size: <integer>, //For exclusively regression-based implementation, should remain 1

    activation: <string> //Activation for the output layer, same options as activation_type
  }, 

  epochs: <integer>, //Number of times the model runs through every observation

  batch_size: <integer>, //Interval, number of forward passes before applying delta gradients

  tt_split: <float>, //Value from 0-1, represents the percentage of the data reserved for testing

  random_seed: <integer>, //Number that determines the sequence of random generated values 

  metric_report_interval: <integer> //Interval, number of observations between the calculation and 
  storage of the training set's RMSE, MAE, and R2.
});
```

### Importing/setting features
Firstly, make sure to store your input features as either a JavaScript object with string labels as the keys and arrays as the values or a .csv file stored as a variable.

_Regardless of the import method, all features must have an equal number of observations, and their target feature must be labeled TARGET in all caps._

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
import {rawCsvData} from '/path-to-file.js'

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

Feature maximums and minimums are stored during the setting methods. The only method that must be called to normalize all features is:

```
benutils.normalize_features(model);
```

### Training and Testing Models

To run your model through its training procedures and store its performance metrics, call:
```
model.train();
```

To run your model through its testing set and store its testing performance, call:
```
model.test();
```

### Predicting Single Observations

To use a trained model to predict a single observation, call:
```
model.predict({
  "FEAT1":<numerical value>,
  "FEAT2":<numerical value>,
  "FEAT3":<numerical value>,
  ...
});  
```
The method returns an array with the predicted value as the first element and the layer values that led to it. 
_(Be aware that model.properties.layers can easily become a large object to return for many consecutive predictions on larger models)_

### Model Evaluation & Performance Metrics

Every training and testing method call stores train/test performance metrics within the model's data. Specifically, performance metrics are stored in:
```
model.properties.performance_metrics.training //.loss_list .mse_list .rmse_list .mae_list .r2
OR
model.properties.performance_metrics.testing //.loss .mse .rmse .mae .r2
```
_to differentiate them in reporting, loss refers to normalized error, while mse refers to the denormalized error_








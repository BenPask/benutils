import * as benlib from './benlib.js';

//CSV DATA IMPORT
// A GIVEN ROW: [0]CRIM, [1]ZN, [2]INDUS, [3]CHAS, [4]NOX, [5]RM, [6]AGE, [7]DIS, [8]RAD, [9]TAX, [10]PTRATIO, [11]B, [12]LSTAT, [13]MEDV


//SELECT FEATURES BY TYPING NAME:VALUE IN GIVEN ROW
const feature_selection = {
    "TARGET":13,
    //put features below
    "RM":5,
    "AGE":6,
    "TAX":9,
    "LSTAT":12,
    "CRIM":0
}

//will store all data as: features{"AGE":[424,1313,1313,134], "TAX":[7546,234234,2342,231]}

//PULL DATA VALUES AND FIND MAX AND MIN stored in maxMins

//Normalize all of the features and store them in dictionary named normalized

//[[1, 2, 3],[4,5,6],[7, 8, 9]]
const mat = [[4,4],[5,5],[6,6]];

let  arr = [[1,2], [1,2], [1,2]];

console.log("cpo2 test", benlib.closest_pow_of_two(2150));

const model = benlib.neuralnet({
    hidden_layer_info: {count:2, init_size:32, style:"po2"},
    activation_type:"relu",
    learn_rate:.01,
    output_layer_info:{init_size:1, activation:"none"},
    epochs:50,
    tt_split:0.25
});

benlib.set_features(model, rawCsvData, feature_selection)
benlib.normalize_features(model)

//model.train({"RM":normalized["RM"],"AGE":normalized["AGE"],"TAX":normalized["TAX"],"LSTAT":normalized["LSTAT"],"CRIM":normalized["CRIM"]},{"TARGET":normalized["TARGET"]});
model.train();
model.test()

console.log(model.layers);
console.log(model.properties.performance_metrics);
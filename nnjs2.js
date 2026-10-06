import * as benlib from './benlib.js';

//CSV DATA IMPORT
// A GIVEN ROW: [0]CRIM, [1]ZN, [2]INDUS, [3]CHAS, [4]NOX, [5]RM, [6]AGE, [7]DIS, [8]RAD, [9]TAX, [10]PTRATIO, [11]B, [12]LSTAT, [13]MEDV
const csvData = rawCsvData.trim().split('\n').map(row=>row.split(','));

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
const features = {}

for(const feat of Object.keys(feature_selection)){
    features[feat] = []
}

const maxMins = {}

//Initialize map of feature maximum and minimum values
for(const feat of Object.keys(feature_selection)){
    const first_val = parseFloat(csvData[0][feature_selection[feat]]);

    maxMins[feat] = {"max":first_val,"min":first_val};
}

//normalized data container
const normalized = {};


function normalize_feature(feat_name){
    //Create a key for this feature in normalized_features and assign the normalized values to it
    normalized[feat_name] = benlib.normalize_array(features[feat_name], maxMins[feat_name]["max"], maxMins[feat_name]["min"]);
};





//PULL DATA VALUES AND FIND MAX AND MIN stored in maxMins
for(const row of csvData){
    for(const feat of Object.keys(feature_selection)){

        const value = parseFloat(row[feature_selection[feat]]);

        features[feat].push(value);

        if(value>maxMins[feat]["max"]) maxMins[feat]["max"] = value;
        if(value<maxMins[feat]["min"]) maxMins[feat]["min"] = value;
    }
}

//Normalize all of the features and store them in dictionary named normalized
for(const feat_name of Object.keys(feature_selection)){
    normalize_feature(feat_name);
}

console.log(normalized);

//[[1, 2, 3],[4,5,6],[7, 8, 9]]
const mat = [[4,4],[5,5],[6,6]];

let  arr = [[1,2], [1,2], [1,2]];

console.log("subtraction test: ", benlib.matsub(mat, arr))



const model = benlib.neuralnet({
    hidden_layer_info: {count:5, init_size:5, style:"increasing"},
    activation:"relu",
    learn_rate:0.001,
    output_layer_info:{init_size:1, activation:"none"},
    epochs:1,
    tt_split:0.25
});

model.train({"RM":normalized["RM"],"AGE":normalized["AGE"],"TAX":normalized["TAX"],"LSTAT":normalized["LSTAT"],"CRIM":normalized["CRIM"]},{"TARGET":normalized["TARGET"]});
model.test()

console.log(model.layers);

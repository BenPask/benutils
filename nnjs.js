//CSV DATA IMPORT
// A GIVEN ROW: [0]CRIM, [1]ZN, [2]INDUS, [3]CHAS, [4]NOX, [5]RM, [6]AGE, [7]DIS, [8]RAD, [9]TAX, [10]PTRATIO, [11]B, [12]LSTAT, [13]MEDV
const csvData = rawCsvData.trim().split('\n').map(row=>row.split(','));

//INPUT LAYER & TARGET INITIALIZATION
const prenormalizedData = [];

const dataIn = [];

const prenormalizedTarget = [];

const targetVar = [];

//WRITE THE ABBREVIATIONS YOUD LIKE TO USE FOR EACH FEATURE, BE CONSISTENT FOR SELECTION
const feature_abbreviations = ["RM", "AGE", "TAX"]

//SELECT FEATURES BY TYPING NAME:VALUE IN GIVEN ROW
const feature_selection = {
    "num_of_features":3,
    "TARGET":13,
    //put features below
    "RM":5,
    "AGE":6,
    "TAX":9
}


//Sets the default minMax values for each feature as well as target
minMax = {"TARGET" : {"min":null,"max":null}}
for(const abbr of feature_abbreviations){
    minMax[abbr] = {"min":null,"max":null}
}

function isMax(valueIn, keyIn){
    if(minMax[keyIn]["max"]!=null){
        if(valueIn>minMax[keyIn]["max"]){
            minMax[keyIn]["max"] = valueIn;  
        }
        
    } 
    else{
        minMax[keyIn]["max"] = valueIn;
    }
}

function isMin(valueIn, keyIn){
    if(minMax[keyIn]["min"]!=null){
        if(valueIn<minMax[keyIn]["min"]){
            minMax[keyIn]["min"] = valueIn;
        }
    }
    else{
        minMax[keyIn]["min"] = valueIn;
    }

}



//Go through and find the max and min of every feature before loading data in (needed for normalization)
for(const row of csvData){
    //input data imports into a json
    const featureRawDataImport = {}

    const RM = parseFloat(row[5]);
    const AGE = parseFloat(row[6]);
    const TAX = parseFloat(row[9]);

    for(const abbr of feature_abbreviations){
        featureRawDataImport[abbr]=row[feature_selection[abbr]]
    }

    //target var data import
    const TARGET = parseFloat(row[feature_selection["TARGET"]]);

    for(const abbr of feature_abbreviations){
        isMin(featureRawDataImport[abbr],abbr);
    }

    //target var min/max finding
    isMin(TARGET, "TARGET");
    isMax(TARGET, "TARGET");

    const prenormalizedListOut = [];
    
    for(const abbr of feature_abbreviations){
        prenormalizedListOut.push(featureRawDataImport[abbr])
    }


    prenormalizedData.push(prenormalizedListOut);
    prenormalizedTarget.push(TARGET);
}

//--normalize data--
//input vars
for(const row of prenormalizedData){
    let nRM = row[0];
    let nAGE = row[1];
    let nTAX = row[2];

    //THIS NEXT!!!!!!!!!!!!!!!!!!!!!!!!!

    nRM = (nRM-minMax["RM"]["min"])/(minMax["RM"]["max"]-minMax["RM"]["min"]);
    nTAX = (nTAX-minMax["TAX"]["min"])/(minMax["TAX"]["max"]-minMax["TAX"]["min"]);
    nAGE = (nAGE-minMax["AGE"]["min"])/(minMax["AGE"]["max"]-minMax["AGE"]["min"]);

    dataIn.push([nRM, nTAX, nAGE]);
}
//target var
for(const val of prenormalizedTarget){
    let nMEDV = val

    nMEDV = (nMEDV-minMax["TARGET"]["min"])/(minMax["TARGET"]["max"]-minMax["TARGET"]["min"]);

    targetVar.push(nMEDV);
}
console.log(targetVar)

//HIDDEN LAYER INITIALIZATION
const numOfHiddenLayers = 2;

const neuronsPerHiddenLayer = 4;

//WEIGHT GENERATION
function generateWeight() {
    //b in the equation, constant variable to adjust variance of starting weights
    const sensitivity = 0.1

    //U[0,1] in the equation, generates a random number from 0 to 1 inclusive on both ends
    const zeroToOne = Math.random() / (1- Number.EPSILON);

    const weightOut = (zeroToOne*(2*sensitivity))-sensitivity;

    return weightOut;
}

//LAYER, WEIGHT, & BIAS INITIALIZATION
const layerWeights = {};

let previousInitLayer = dataIn[0];

const layerBiases = {};

const defaultBias = 0;

for(let i = 0; i < numOfHiddenLayers; i++){
    layerWeights[`hidden_layer_${i}`] = []
    layerBiases[`hidden_layer_${i}`] = []

    for(let j = 0; j < neuronsPerHiddenLayer; j++){
        layerWeights[`hidden_layer_${i}`].push([]);
        layerBiases[`hidden_layer_${i}`].push(defaultBias);
        for(let k = 0; k < previousInitLayer.length; k++){
            layerWeights[`hidden_layer_${i}`][j].push(generateWeight());
        }
    }
    previousInitLayer = layerWeights[`hidden_layer_${i}`];
    
}

//Output layer weight and bias init
layerWeights['output_layer'] = [[]];
layerBiases['output_layer'] = [defaultBias];


for(let i = 0; i < previousInitLayer.length; i++){
    layerWeights['output_layer'][0].push(generateWeight())
}

function neuronWeightedSum(valuesIn, weightsIn){
    
    const listOutput = [];
    let sumOutput = 0;
    

    //for each value in valuesIn (the input numbers)
    for(let i = 0; i < valuesIn.length; i++){
        listOutput.push(valuesIn[i]*weightsIn[i]);
    }

    for(const value in listOutput){
        sumOutput += listOutput[value];
    }

    
    return sumOutput;
}

function layerWeightedSum(valuesIn, layerIn){

    const layerWeightedSumOut = [];
    
    //for each neuron in layerIn
    for(let j = 0; j < layerIn.length; j++){
        const neuronOut = neuronWeightedSum(valuesIn,layerIn[j]);
        layerWeightedSumOut.push(neuronOut);
    }
    
    return layerWeightedSumOut;
}

function applyBias(vectorIn, indexOfLayer){
    const layerSumBiasedOutput = [];

    for(let i = 0; i < vectorIn.length; i++){
        layerSumBiasedOutput.push(vectorIn[i]+layerBiases[`hidden_layer_${indexOfLayer}`][i])
    }

    return layerSumBiasedOutput;
}

function applyRelu(vectorIn){
    const layerPostReluOutput = [];

    for(let i = 0; i < vectorIn.length; i++){

        if(vectorIn[i] < 0){
            layerPostReluOutput.push(0)
        }
        else{
            layerPostReluOutput.push(vectorIn[i])
        }   
    }

    return layerPostReluOutput;
}

for(const instance of dataIn){
    const inputVector = instance;
    
    let prevOutput = inputVector;

    for(let curHiddenLayer = 0; curHiddenLayer < numOfHiddenLayers; curHiddenLayer++){
        const weightedSum = layerWeightedSum(prevOutput, layerWeights[`hidden_layer_${curHiddenLayer}`]);
        const biasedOut = applyBias(weightedSum, curHiddenLayer);
        const reluOut = applyRelu(biasedOut);

        prevOutput = reluOut;
    }


    layerWeightedSum(prevOutput, layerWeights['output_layer'])
}












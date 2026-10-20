import {neuralnet, set_features, set_features_csv, normalize_features} from "../src/network.js";
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

const model = neuralnet({
    hidden_layer_info: {
        count:2, 
        init_size:64, 
        style:"po2"
    },
    activation_type:"relu",
    learn_rate:.01,
    output_layer_info:{
        init_size:1, 
        activation:"none"
    },
    epochs:100,
    tt_split:0.25,
    random_seed:58373
});

set_features_csv(model, rawCsvData, feature_selection)
normalize_features(model)

model.train();
model.test();

test('feature_lengths', () => {
    expect(model.properties.features["RM"].length).toBe(model.properties.features["TARGET"].length);
});

/*
test('predictions_and_targets', () => {
    expect(model.properties.predictions_and_targets.training.predictions.length).toBe(50);
    expect(model.properties.predictions_and_targets.testing.predictions.length).toBe(126);
});
*/
/*
test('predict', () => {
    expect(model.predict({
    "RM":6.5750,
    "AGE":65.20,
    "TAX":296.0,
    "LSTAT":4.98,
    "CRIM":0.00632})).toBeDefined();
});

test('predict-layer-logging', () => {
    expect(model.predict({
    "RM":6.5750,
    "AGE":65.20,
    "TAX":296.0,
    "LSTAT":4.98,
    "CRIM":0.00632})[1]['output_layer']['output'][0] == model.predict({"RM":13.5750,
    "AGE":85.20,
    "TAX":299.0,
    "LSTAT":6.98,
    "CRIM":0.00664})[1]['output_layer']['output'][0]).toBe(false);
});

*/

test('metric reporting', () => {
    expect(model.properties.performance_metrics.testing.mse).toBeDefined();
    console.log(model.properties.performance_metrics)
});


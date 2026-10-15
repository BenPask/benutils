import {neuralnet, set_features, set_features_csv, normalize_features} from "../src/network.js";
import {rawCsvData} from './housing.js'



const feature_selection = {
    "TARGET":13,
    //put features below
    "RM":5,
    "AGE":6,
    "TAX":9,
    "LSTAT":12,
    "CRIM":0
}

const model = neuralnet({
    hidden_layer_info: {count:11, init_size:2, style:"hourglass", hourglass_operation:"*"},
    activation_type:"tanh",
    learn_rate:.01,
    output_layer_info:{init_size:1, activation:"none"},
    epochs:50,
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

test('predictions_and_targets', () => {
    expect(model.properties.predictions_and_targets.training.predictions.length).toBe(50);
    expect(model.properties.predictions_and_targets.testing.predictions.length).toBe(126);
});

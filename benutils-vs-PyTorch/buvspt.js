import {neuralnet, set_features, set_features_csv, normalize_features} from "../src/network.js";

const model = neuralnet({
    hidden_layer_info: {
        count:1, 
        init_size:2, 
        style:"flat"
    },
    activation_type:"relu",
    learn_rate:.01,
    output_layer_info:{
        init_size:1, 
        activation:"none"
    },
    epochs:1,
    tt_split:1,
    random_seed:58373
});

simple_input = torch.tensor([[3.0, 5.0], [12.0, 15.0], [25.0, 35.0]])
simple_targets = torch.tensor([[22.0], [40.0], [11.0]])

const features = {
    "feat1":[3.0, 12.0, 25.0],
    "feat2":[5.0, 15.0, 35.0]
}

model.set_features(features)
model.train()
model.test()

console.log(model.layers)
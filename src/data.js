import {zto_random, round_float} from './math.js';

//-------DATA-------
export function normalize_array(arr, max_in, min_in){
    const normalized_arr = []

    for(const value of arr){
        const normalized_value = (value-min_in)/(max_in-min_in);
        normalized_arr.push(normalized_value);
    }

    return normalized_arr;
}

export function denormalize_array(arr, max_in, min_in){
    const denormalized_arr = []

    for(const value of arr){
        const denormalized_value = (value*(max_in-min_in))+min_in;
        denormalized_arr.push(denormalized_value);
    }

    return denormalized_arr;
}

export function shuffle_feature_data(features_in, model=null){

    const feat_names = Object.keys(features_in)

    const observation_count = features_in[feat_names[0]].length;

    const features_copy = {};
    for(const name of feat_names){
        features_copy[name] = [...features_in[name]];
    }

    //fisher yates shuffle
    for(let i = observation_count - 1; i > 0; i--){

        let zto_gen = zto_random();

        if(model!=null) zto_gen = zto_random(model);

        const j = round_float((zto_gen * (i+1))-0.5);

        for(const name of feat_names){
            const temp = features_copy[name][i];
            features_copy[name][i] = features_copy[name][j];
            features_copy[name][j] = temp;
        }
    }

    return features_copy;
}
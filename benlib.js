//-------MATH-------
const euler = 2.718281828459045;
let global_seed = 75128;


export function zto_random(model=null){
    const a = 49234;
    const m = 1000000007; //large prime number

    if(model!=null){
        model.properties.current_seed = (model.properties.current_seed * a) % m;
        return model.properties.current_seed / m;
    }
    //calc next seed
    global_seed = (global_seed*a)%m;

    //turn it into a number between 0 and 1
    return global_seed/m;
}

export function xpo(value, power){
    let effective_power = power;

    if(power<0){
        effective_power = power * -1;
    }
    
    let current = value;
    for(let i = 0; i<effective_power-1; i++){
        current = current*value;
    }
    
    if(power<0){
        current = 1/current;
    }
    
    return current;
}

export function round_float(float){
    if(float<0){
        return -1*(((-1*float)+0.5) | 0);
    }
    
    return float+0.5 | 0;
}

export function absolute_value(value){
    if(value<0){
        return value * -1
    }
    return value
}

export function sqroot(value, tolerance=1e-15){
    if(value<0){
        throw new Error("Cannot take the square root of negative numbers")
    }
    if(value == 0){
        return 0
    }

    let guess = value/2
    
    let iter = 0
    while(iter < 1000){
        let next_guess = 0.5*(guess + value/guess)

        if(absolute_value(guess - next_guess) < tolerance){
            return next_guess
        }

        guess = next_guess
        iter += 1;
    }
    return guess;
}

export function closest_pow_of_two(val){

    if(typeof val != 'number'){
        throw new Error("non-number attempted to pass through closeset_pow_of_two");
    }

    let current = null;
    let prev_current = 2;

    while(true){
        current = prev_current*2;

        if(current>val){
            break;
        }
        prev_current = current;
    }

    if(absolute_value(val-prev_current)>absolute_value(val-current)){
        return current;
    }
    else{
        return prev_current;
    }
}

//-------ARRAYS & MATRICES-------
export function array_to_matrix(array_in){
    if(array_in.length == 0) return [[]];

    let array_out = array_in;

    //dealing with 1d array cases
    if(!Array.isArray(array_in[0])){
        array_out = [array_in];
    }

    return array_out;
}

export function array_mean(array_in){
    let total = 0;
    for(const value of array_in){
        total += value;
    }
    return total/array_in.length;
}

export function arrmult(array1, array2){
    let array_out = [];
    for(let i = 0; i<array1.length; i++){
        array_out.push(array1[i]*array2[i]);
    }
    return array_out;
}

export function transpose_matrix(m){

    let output = [];

    if(!Array.isArray(m[0])){
        return m.map(x => [x])
    }
    //var for # of rows
    const m_rows = m.length;

    //var for # of columns
    const m_cols = m[0].length;
    
    //for # of columns in matrix
    for(let c = 0; c<m_cols; c++){
        const cur_row_out = [];
        //for # of rows in matrix
        for(let r = 0; r<m_rows; r++){
            cur_row_out.push(m[r][c]);
        }
        output.push(cur_row_out);
    }

    
    return output;
}

export function matmult(m1, m2){
    
    //transpose m2 for easier multiplication
    m2 = transpose_matrix(m2);

    m1 = array_to_matrix(m1);
    m2 = array_to_matrix(m2);
    
    let m1_rows = m1.length;
    let m1_cols = m1[0].length;

    let m2_rows = m2.length;
    let m2_cols = m2[0].length;

    if(m1[0].length!=m2[0].length){
        throw new Error("The amount of columns in m1 must match the amount of rows in m2");
    }

    let product = [];

    //for the # of rows in the first matrix
    for(let r = 0; r<m1_rows; r++){
        product.push([]);
    }
    
    for(let r_m1 = 0; r_m1<m1_rows; r_m1++){
        //for the # of columns in the transposition of m2
        for(let r_m2 = 0; r_m2<m2_rows; r_m2++){
            let cur_value_out = 0;
            //for the # of columns in the first matrix
            for(let c_m1 = 0; c_m1<m1_cols; c_m1++){
                cur_value_out += m1[r_m1][c_m1]*m2[r_m2][c_m1];
            }
            product[r_m1].push(cur_value_out);
        }
    }
    return product;
}

export function matadd(matrix1, matrix2){
    let matrix_out = [];

    //deal with 1d array input cases (just formatting to [[m1]] instead of [m1])
    matrix1 = array_to_matrix(matrix1)
    matrix2 = array_to_matrix(matrix2)
    

    for(let r = 0; r<matrix1.length; r++){
        matrix_out.push([])
        for(let c = 0; c<matrix1[0].length; c++){
            matrix_out[r].push(matrix1[r][c]+matrix2[r][c])
        }
    }

    return matrix_out;
}

export function matsub(matrix1, matrix2){
    matrix1 = array_to_matrix(matrix1)
    matrix2 = array_to_matrix(matrix2)

    const negative_matrix2 = [];

    for(const array of matrix2){
        negative_matrix2.push(array.map(x => x*(-1)));
    }

    const matrix_out = matadd(matrix1, negative_matrix2);
    return matrix_out;
}

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

//-------NEURAL_NETWORK-------
function generate_weight(model, prev_layer_size){
    //Utilizing he weight generation to prevent explosions to infinity
    const limit = sqroot(6/prev_layer_size);

    const zeroToOne = zto_random(model);

    const weightOut = (zeroToOne*2*limit)-limit;

    return weightOut;
}

export function set_features(model, features_in){
    const maxMins = model.properties.feature_data.maxMins;
    const features = {};

    for(const feat_name of Object.keys(features_in)){
        const first_val = features_in[feat_name][0];

        maxMins[feat_name] = {"max":first_val,"min":first_val};
    }

    for(const feat_name of Object.keys(features_in)){
        for(const value of features_in[feat_name]){
            features[feat_name].push(value);

            if(value>maxMins[feat_name]["max"]) maxMins[feat_name]["max"] = value;
            if(value<maxMins[feat_name]["min"]) maxMins[feat_name]["min"] = value;
        }
    }
    features = shuffle_feature_data(features, model);
    model.properties.features = features;
}

export function set_features_csv(model, csvData, feature_selection){
    csvData = csvData.trim().split('\n').map(row=>row.split(','));

    //csv header safegaurd, default=0 or "no headers"
    let start_index = 0;

    const maxMins = model.properties.feature_data.maxMins
    let features = {};

    for(const feat_name of Object.keys(feature_selection)){
        features[feat_name] = [];
    }

    //safegaurd for csv headers, just removing them because feature_selection handles labels
    if(isNaN(parseFloat(csvData[0][feature_selection["TARGET"]]))){
        start_index = 1;
    }

    //Initialize map of feature maximum and minimum values
    for(const feat_name of Object.keys(feature_selection)){
        const first_val = parseFloat(csvData[start_index][feature_selection[feat_name]]);

        maxMins[feat_name] = {"max":first_val,"min":first_val};
    } 

    for(let row = start_index; row < csvData.length; row++){
        for(const feat_name of Object.keys(feature_selection)){

            const value = parseFloat(row[feature_selection[feat_name]]);

            
            features[feat_name].push(value);

            if(value>maxMins[feat_name]["max"]) maxMins[feat_name]["max"] = value;
            if(value<maxMins[feat_name]["min"]) maxMins[feat_name]["min"] = value;
        }
    }

    features = shuffle_feature_data(features, model);
    model.properties.features = features;
}

export function normalize_features(model){
    const features = model.properties.features;

    for(const feat_name of Object.keys(features)){
        normalize_feature(model, feat_name);
}
}

export function normalize_feature(model, feat_name){
    const maxMins = model.properties.feature_data.maxMins;
    const features = model.properties.features;
    //Create a key for this feature in normalized_features and assign the normalized values to it
    features[feat_name] = normalize_array(features[feat_name], maxMins[feat_name]["max"], maxMins[feat_name]["min"]);
};

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

function fwd_pass(model, layer_index){
    const indexed_layers = Object.values(model.layers);

    const prev_layer_output = indexed_layers[layer_index-1]["output"];
    const weight_matrix = indexed_layers[layer_index]["main"]["weights"];
    const bias_arr = indexed_layers[layer_index]["main"]["biases"];

    const t_prev_layer_output = transpose_matrix(prev_layer_output);

    const pre_t_multiplied_output = matmult(weight_matrix, t_prev_layer_output);
    
    const multiplied_output = transpose_matrix(pre_t_multiplied_output);
    
    let arr_out = [];
    
    for(let i = 0;i<multiplied_output[0].length;i++){
        arr_out.push(multiplied_output[0][i]+bias_arr[i]);
    }
    
    indexed_layers[layer_index]["preactivation"] = arr_out;

    //dynamic activation function based on model.properties.activation_type
    if(layer_index == indexed_layers.length-1){
        arr_out = activate(model, arr_out, true);
    }
    else{
        arr_out = activate(model, arr_out);
    }
    
    indexed_layers[layer_index]["output"] = arr_out;
}

function calculate_mse(predictions, targets){
    let total_error = 0;

    for(let i = 0; i < predictions.length; i++){
        let error = targets[i] - predictions[i];
        total_error += error*error;
    }
    
    return total_error/predictions.length;
}

function calculate_rmse(model, predictions, targets){
    const prediction_count = predictions.length;

    let total_residuals = 0;
    const targets_og_min = model.properties.feature_data.maxMins["TARGET"]["min"];
    const targets_og_max = model.properties.feature_data.maxMins["TARGET"]["max"];

    predictions = denormalize_array(predictions, targets_og_max, targets_og_min);
    targets = denormalize_array(targets, targets_og_max, targets_og_min);

    for(let i = 0; i < prediction_count; i++){
        let residual = (targets[i]-predictions[i])*(targets[i]-predictions[i]);
        total_residuals += residual;
    }

    return(sqroot(total_residuals/prediction_count));
}

function calculate_mae(model, predictions, targets){
    const prediction_count = predictions.length;

    let total_error = 0;

    const targets_og_min = model.properties.feature_data.maxMins["TARGET"]["min"];
    const targets_og_max = model.properties.feature_data.maxMins["TARGET"]["max"];

    predictions = denormalize_array(predictions, targets_og_max, targets_og_min);
    targets = denormalize_array(targets, targets_og_max, targets_og_min);

    for(let i = 0; i < prediction_count; i++){
        let error = absolute_value(targets[i]-predictions[i]);
        total_error += error;
    }

    return total_error/prediction_count;
}

function calculate_rsquared(model, predictions, targets){
    const prediction_count = predictions.length;

    let sum_residuals = 0;
    let sum_squares = 0;

    const target_mean = array_mean(targets);

    for(let i = 0; i < prediction_count; i++){
        sum_residuals += (targets[i]-predictions[i])*(targets[i]-predictions[i]);
        sum_squares += (targets[i]-target_mean)*(targets[i]-target_mean);
    }

    return 1-(sum_residuals/sum_squares);
}

function calculate_loss_gradient(model, target_value, layer_index){
    const indexed_layers = Object.values(model.layers);

    if(layer_index==indexed_layers.length-1){
        const loss_gradient = [];
        const post_activation_final_output = model.layers["output_layer"]["output"];

        for(const value of post_activation_final_output){
        //CHANGE LATER if no longer regression
        loss_gradient.push(value-target_value);
        }
        return loss_gradient;
    }

    const next_weights = indexed_layers[layer_index+1]["main"]["weights"];
    const next_delta = indexed_layers[layer_index+1]["delta"]["layer_delta"];


    const transposed_next_weights = transpose_matrix(next_weights);

    const transposed_next_delta = transpose_matrix(next_delta);

    let loss_gradient = matmult(transposed_next_weights, transposed_next_delta);

    loss_gradient = transpose_matrix(loss_gradient);

    return loss_gradient[0];
}

function calculate_derivative_array(model, layer_index){

    const indexed_layers = Object.values(model.layers);

    if(layer_index==indexed_layers.length-1){
        const pre_activation_output = model.layers["output_layer"]["preactivation"];
    
        const derivative_array = deriv_activate(model, pre_activation_output, true);
    
        return derivative_array;
    }

    
    const pre_activation_output = indexed_layers[layer_index]["preactivation"];

    const derivative_array = deriv_activate(model, pre_activation_output);

    return derivative_array;
}

function bwd_pass(model, layer_index, target_value){
    const indexed_layers = Object.values(model.layers);

    const loss_gradient = calculate_loss_gradient(model, target_value, layer_index);


    const derivative_array = calculate_derivative_array(model, layer_index);
  

    const layer_delta = arrmult(loss_gradient, derivative_array);
    indexed_layers[layer_index]["delta"]["layer_delta"] = layer_delta;

    const previous_layer_output = indexed_layers[layer_index-1]["output"];
    

    const delta_col = transpose_matrix(layer_delta);

    const delta_weight_gradient = matmult(delta_col, previous_layer_output);

    

    const current_delta_gradient = indexed_layers[layer_index]["delta"]["weights"];
    const current_bias_array = indexed_layers[layer_index]["delta"]["biases"];

    
    indexed_layers[layer_index]["delta"]["weights"] = matadd(current_delta_gradient, delta_weight_gradient);
    for(let i = 0; i < layer_delta.length; i++){
        current_bias_array[i] += layer_delta[i];
    }

}

function average_deltas(model, custom=0){

    const learn_rate = model.properties.learn_rate;
    
    const indexed_layers = Object.values(model.layers);

    for(let i = 1; i<indexed_layers.length; i++){
        let current_delta_weights = indexed_layers[i]["delta"]["weights"];
        const current_delta_biases = indexed_layers[i]["delta"]["biases"];
        let interval_divisor = model.properties.batch_size;


        if(custom!=0){
            interval_divisor = custom;
        }
        
        let averaged_delta_weights = [];

        current_delta_weights = array_to_matrix(current_delta_weights);
        for(const array of current_delta_weights){
            const row_out = [];
            for(const value of array){
                row_out.push((value/interval_divisor)*learn_rate);
            }
            averaged_delta_weights.push(row_out);
        }

        const averaged_delta_biases = [];

        for(const value of current_delta_biases){
            averaged_delta_biases.push((value/interval_divisor)*learn_rate);
        }

        indexed_layers[i]["delta"]["weights"] = averaged_delta_weights;
        indexed_layers[i]["delta"]["biases"] = averaged_delta_biases;
    }
}

function update_weights_and_biases(model){
    const indexed_layers = Object.values(model.layers);

    for(let i = 1; i<indexed_layers.length; i++){
        const current_weights = indexed_layers[i]["main"]["weights"];
        const current_delta_weights = indexed_layers[i]["delta"]["weights"];

        const current_biases = indexed_layers[i]["main"]["biases"];
        const current_delta_biases = indexed_layers[i]["delta"]["biases"];

        indexed_layers[i]["main"]["weights"] = matsub(current_weights, current_delta_weights);
        indexed_layers[i]["main"]["biases"] = matsub(current_biases, current_delta_biases)[0];        
    }
}

function reset_deltas(model){
    const indexed_layers = Object.values(model.layers);

    for(let i = 1; i<indexed_layers.length; i++){
        const reset_delta_weights = [];
        const reset_delta_biases = [];

        const current_delta_biases = indexed_layers[i]["delta"]["biases"];
        const current_delta_weights = indexed_layers[i]["delta"]["weights"];

        for(const array of current_delta_weights){
            const row_out = [];
            for(const value of array){
                row_out.push(0);
            }
            reset_delta_weights.push(row_out);
        }

        for(const value of current_delta_biases){
            reset_delta_biases.push(0);
        }
        indexed_layers[i]["delta"]["weights"] = reset_delta_weights;
        indexed_layers[i]["delta"]["biases"] = reset_delta_biases; 
    }
}

function activate(model, arr_in, o_layer=false){
    if(o_layer == true){
        const acitvation_func = activation_map[model.properties.output_layer_info.activation];
        return acitvation_func(arr_in);
    }
    const acitvation_func = activation_map[model.properties.activation_type];
    return acitvation_func(arr_in);
}

function deriv_activate(model, arr_in, o_layer=false){
    if(o_layer == true){
        const acitvation_func = deriv_acitvation_map[model.properties.output_layer_info.activation];
        return acitvation_func(arr_in);
    }
    const acitvation_func = deriv_acitvation_map[model.properties.activation_type];
    return acitvation_func(arr_in);
}

function flat_build(model, num_of_layers){

    const init_value = model.properties.hidden_layer_info.init_size;

    const schema_values = [];

    for(let i = 0; i < num_of_layers; i++){
        schema_values.push(0);
    }
    
    for(let i = 0; i < schema_values.length; i++){
        schema_values[i] = init_value;
    }
    

    return schema_values;
}

function power_of_two_build(model, num_of_layers){

    let init_value = closest_pow_of_two(model.properties.hidden_layer_info.init_size);
    //How many incremental divisions by 2 will it take to get down to just 2 neurons
    let max_increments = 0;
    let temp_val = init_value;

    while(temp_val > 2){
        temp_val = temp_val/2;
        max_increments += 1;
    }

    let layers_per_increment = round_float((num_of_layers/max_increments)-0.5);

    if(layers_per_increment == 0){
        layers_per_increment = 1;
    }

    const schema_values = [];
    
    for(let i = 0; i < num_of_layers; i++){
        schema_values.push(0);
    }

    let cur_increment_count = 0;
    
    for(let i = 0; i < schema_values.length; i++){

        cur_increment_count++;
        
        schema_values[i] = init_value;

        if(cur_increment_count>=layers_per_increment){
            if(init_value/2 >= 2){
                init_value = init_value/2;
                cur_increment_count = 0;
            }
        }
    }
    return schema_values;
}

function hourglass_value(model, value, inverse=false){
    let hourglass_operator = model.properties.hidden_layer_info.hourglass_operation;

    const operation_map = {"/":"*", "sqroot":"**", "*":"/", "**":"sqroot"};

    if(inverse){
        hourglass_operator = operation_map[hourglass_operator];
    }
    
    switch (hourglass_operator) {
        case "/":
            return value/2;
        case "*":
            return value*2;
        case "sqroot":
            return sqroot(value);
        case "**":
            return value**2;
        default:
            return value/2;
    }
}

function hourglass_build(model, num_of_layers){
    let init_value = model.properties.hidden_layer_info.init_size;

    const schema_values = [];

    for(let i = 0; i < num_of_layers; i++){
        schema_values.push(0);
    }

    for(let i = 0; i < num_of_layers; i++){
        if(i<round_float(num_of_layers/2) && (!(num_of_layers%2!=0 && i==round_float((num_of_layers/2)-0.5)))){
            schema_values[i] = init_value;
            if(hourglass_value(model, init_value) >= 2){
                init_value = round_float(hourglass_value(model, init_value));
            }
        }
        else{
            schema_values[i] = init_value;
            if(hourglass_value(model, init_value) >= 2){
                init_value = round_float(hourglass_value(model, init_value, true));
            }
        }
    }
    return schema_values;
}

function build_matrices(model){
    model.layers["input_layer"] = {"output":[]};
    for(let i = 1;i<=model.properties.hidden_layer_info.count;i++){
        model.layers[`hidden_layer_${i}`] = {"main":{"weights":[],"biases":[]}, "delta":{"weights":[],"biases":[]},"preactivation":[], "output":[]};
    }
    model.layers["output_layer"] = {"main":{"weights":[],"biases":[]}, "delta":{"weights":[],"biases":[]}, "preactivation":[], "output":[]};

    //we have 1D arrays for the main & delta weights, we now add an amount of arrays(rows of matrix) for the size of the layer
    //This next line just lets us call the layers in order via index. 0 is the input layer
    const indexed_layers = Object.values(model.layers);

    const style_map = {"flat":flat_build,"po2":power_of_two_build,"hourglass":hourglass_build};

    const hidden_count = model.properties.hidden_layer_info.count;
    const hidden_layer_schema = style_map[model.properties.hidden_layer_info.style](model, hidden_count);

    //matrix builder, does not yet contain columns or weights ie: layers["hidden_layer_1"] = [[][][][][]] if hidden layer size was 5
    for(let i=1; i<indexed_layers.length-1; i++){
        const layer_size = hidden_layer_schema[i-1];
        
        for(let r = 0; r<layer_size;r++){
            //add empty rows to hidden layer's main weights
            indexed_layers[i]["main"]["weights"].push([]);

            //add empty rows to the delta version of hidden layer's weights
            indexed_layers[i]["delta"]["weights"].push([]);
        }
    }    
    for(let r = 0; r<model.properties.output_layer_info.init_size;r++){
            //add empty rows to output layer weights
            indexed_layers[indexed_layers.length-1]["main"]["weights"].push([]);

            //add empty rows to delta version of the output layer weights
            indexed_layers[indexed_layers.length-1]["delta"]["weights"].push([]);
    }
}

function fill_matrices(model, feature_count){
    const indexed_layers = Object.values(model.layers);

    //Because the amount of columns(weights) per layer are determined by the output of the previouse layer,
    //initialize the count to be the size of the INPUT layer, then adjust as you move through the layers
    let prev_layer_size = feature_count;

    for(let i=1; i<indexed_layers.length; i++){
        for(const m_row of indexed_layers[i]["main"]["weights"]){
            //push a generated weight for the size of the previous layers output.
            for(let i = 0;i<prev_layer_size;i++) m_row.push(generate_weight(model, prev_layer_size));
        }
        for(const m_row of indexed_layers[i]["delta"]["weights"]){
            //push a 0 for the size of the previous layers output.
            for(let i = 0;i<prev_layer_size;i++) m_row.push(0);
        }
        //update the previous layer size variable to be this layer before moving on
        prev_layer_size = indexed_layers[i]["main"]["weights"].length;

        for(let n = 0; n < indexed_layers[i]["main"]["weights"].length; n++){
            indexed_layers[i]["main"]["biases"].push(0);
            indexed_layers[i]["delta"]["biases"].push(0);
        }
    }    
}

function iteration(model, obs_num, feature_data, target_data, bp=true){
    model.layers["input_layer"]["output"] = [];
    
    const target_value  = target_data["TARGET"][obs_num];

    //set the model.layers["input_layer"]["output"] to the values of the feature's for this observation number
    for(const feat_name of Object.keys(feature_data)){
        model.layers["input_layer"]["output"].push(feature_data[feat_name][obs_num]);
    };

    for(let i = 1;i < Object.keys(model.layers).length; i++){
        fwd_pass(model, i);
    }
    if(bp==true){
        for(let i = Object.keys(model.layers).length-1;i >= 1; i--){
            bwd_pass(model, i, target_value);
        }
    }
    
}

function train(model){
    const loss_list = [];
    const rmse_list = model.properties.performance_metrics.training.rmse_list;
    const mae_list = model.properties.performance_metrics.training.mae_list;

    const target_data = {};
 
    const feature_data = model.properties.features;
    target_data["TARGET"] = model.properties.features.TARGET;

    delete feature_data["TARGET"];

    
    
    const training_observation_count = round_float(target_data["TARGET"].length * (1-model.properties.tt_split));
    const testing_observation_count = target_data["TARGET"].length - training_observation_count;  
    
    //# of features being used NOT including target
    const feature_count = Object.keys(feature_data).length;
    
    const feature_training_data = {};
    const feature_testing_data = {};

    //guardrail to confirm all features including target have the same number of observations
    for(const feat_name of Object.keys(feature_data)){
        if(target_data["TARGET"].length != feature_data[feat_name].length){
            throw new Error("Features and targets are not all the same size!(or wrong input format, etc.)");
        }
    }

    for(const feat_name of Object.keys(feature_data)){
        //Split feature data into training and testing sets
        feature_training_data[feat_name] = feature_data[feat_name].slice(0, training_observation_count);
        feature_testing_data[feat_name] = feature_data[feat_name].slice(training_observation_count, target_data["TARGET"].length);
    }

    const target_training_data = {};
    target_training_data["TARGET"] = target_data["TARGET"].slice(0, training_observation_count);
    const target_testing_data = {};
    target_testing_data["TARGET"] = target_data["TARGET"].slice(training_observation_count, target_data["TARGET"].length);


    model.properties.stored_testing_data.t_feature_data = feature_testing_data;
    model.properties.stored_testing_data.t_target_data = target_testing_data;

    

    //build simple structure without columns or weights, but including rows via the neuron count(size) of each layer
    build_matrices(model);
    fill_matrices(model, feature_count);

    for(let e = 0; e<model.properties.epochs; e++){
    
        let total_epoch_loss = 0;

        let iterations_since_interval = 0;

        const epoch_predictions = [];
        const epoch_targets = [];

        for(let i = 0; i<training_observation_count;i++){
            
            iteration(model, i, feature_training_data, target_training_data);

            //CHANGE if moving away from single output neuron regression model
            epoch_predictions.push(model.layers["output_layer"]["output"][0]);
            epoch_targets.push(target_training_data["TARGET"][i]);

            iterations_since_interval += 1;

            if(iterations_since_interval==model.properties.batch_size){
                iterations_since_interval = 0;
                average_deltas(model);
                update_weights_and_biases(model);
                reset_deltas(model);
            }
            total_epoch_loss += calculate_mse(epoch_predictions, epoch_targets);

            
        }

        if(iterations_since_interval%model.properties.metric_report_interval==0){
            rmse_list.push(calculate_rmse(model, epoch_predictions, epoch_targets));
            mae_list.push(calculate_mae(model, epoch_predictions, epoch_targets));
        }

        loss_list.push(total_epoch_loss/training_observation_count);

        if(iterations_since_interval>0){
            average_deltas(model, iterations_since_interval);
            update_weights_and_biases(model);
            reset_deltas(model);
        }
        
    }
    model.properties.performance_metrics.training.loss_list = loss_list;
}

function test(model){

    const feature_testing_data = model.properties.stored_testing_data.t_feature_data;
    const target_testing_data = model.properties.stored_testing_data.t_target_data;

    const testing_observation_count = target_testing_data["TARGET"].length;

    let total_test_loss = 0;
        const test_predictions = [];
        const test_targets = [];

    for(let i = 0; i<testing_observation_count;i++){
        

        iteration(model, i, feature_testing_data, target_testing_data, false);

        //CHANGE if moving away from single output neuron regression model
        test_predictions.push(model.layers["output_layer"]["output"][0])
        test_targets.push(target_testing_data["TARGET"][i])
    }
    total_test_loss = calculate_mse(test_predictions, test_targets);
    
    const metrics = model.properties.performance_metrics.testing;

    metrics.loss = total_test_loss;
    metrics.rmse = calculate_rmse(model, test_predictions, test_targets);
    metrics.mae = calculate_mae(model, test_predictions, test_targets);
    metrics.r2 = calculate_rsquared(model, test_predictions, test_targets);
    
}

function reluActivation(arr){
    let arrOut = [];
    arrOut = arr.map(x => x > 0 ? x : 0);

    return arrOut
}

function deriv_relu_activation(arr){
    let arrOut = [];

    for(const val of arr){
        if(val <= 0){
            arrOut.push(0);
        }
        else{
            arrOut.push(1);
        }
    }
    return arrOut;
}

function leaky_relu_activation(arr){
    const leaky_constant = 0.01;
    let arrOut = [];

    arrOut = arr.map(x => x > 0 ? x : x*leaky_constant);

    return arrOut;
}

function deriv_leaky_relu_activation(arr){
    const leaky_constant = 0.01
    let arrOut = [];

    for(const val of arr){
        if(val <= 0){
            arrOut.push(leaky_constant);
        }
        else{
            arrOut.push(1);
        }
    }
    return arrOut;
}

function sigmoid_activation(arr){
    let arrOut = [];

    for(const val of arr){
        arrOut.push(xpo(euler, val)/(xpo(euler, val)+1));
    }

    return arrOut;
}

function deriv_sigmoid_activation(arr){
    let arrOut = [];

    for(const val of arr){
        arrOut.push(xpo(euler, val*-1)/xpo((xpo(euler, val*-1)+1), 2));
    }

    return arrOut;
}

function tanh_activation(arr){
    let arrOut = [];
   
    for(const val of arr){
        const numerator = (xpo(euler, val)-xpo(euler, val*-1));
        const denominator = (xpo(euler, val)+xpo(euler, -1*val));
        arrOut.push(numerator/denominator);
    }

    return arrOut;
}

function deriv_tanh_activation(arr){
    let arrOut = [];
   
    for(const val of arr){
        const numerator = (xpo(euler, val)-xpo(euler, val*-1));
        const denominator = (xpo(euler, val)+xpo(euler, -1*val));
        arrOut.push(1-xpo((numerator/denominator), 2));
    }

    return arrOut;
}

function no_activation(arr){
    return arr;
}

function deriv_no_activation(arr){
    return arr.map(x => 1);
}

const activation_map = {"none":no_activation,"relu":reluActivation,"sigmoid":sigmoid_activation,"tanh":tanh_activation,"leaky_relu":leaky_relu_activation};
const deriv_acitvation_map = {"none":deriv_no_activation,"relu":deriv_relu_activation,"sigmoid":deriv_sigmoid_activation,"tanh":deriv_tanh_activation,"leaky":deriv_leaky_relu_activation};

export function neuralnet({
    //default parameters for neurat net initialization
    hidden_layer_info = {count:5, init_size:5, style:"flat", hourglass_operation:"/"},
    activation_type = "relu",
    learn_rate = 0.01,
    output_layer_info = {init_size:1, activation:"none"},
    epochs = 100,
    batch_size = 10,
    tt_split = 0.25,
    random_seed = 58373,
    features = {},
    feature_data = {maxMins:{}},
    stored_testing_data = {t_feature_data:null, t_target_data:null},
    performance_metrics = {training:{loss_list:null, rmse_list:[], mae_list:[]},testing:{loss:null, rmse:null, r2:null, mae:null}},
    normalization_statistics = {},
    metric_report_interval = 10}={}) {

        hidden_layer_info = {
            count:5, 
            init_size:5, 
            style:"flat", 
            hourglass_operation:"/",
            ...hidden_layer_info
        }

        console.log(`now buidling model[SEED: ${random_seed}]
${epochs} epochs
${tt_split} train/test split
${activation_type} activation
${batch_size} batch size
${hidden_layer_info.style} architecture, hg_oper?: ${hidden_layer_info.hourglass_operation}
${learn_rate} learning rate`);

        const model_properties = {
            hidden_layer_info:hidden_layer_info,
            activation_type:activation_type,
            learn_rate:learn_rate,
            output_layer_info:output_layer_info,
            epochs:epochs,
            batch_size:batch_size,
            tt_split:tt_split,
            random_seed:random_seed,
            current_seed:random_seed,
            features:features,
            feature_data:feature_data,
            metric_report_interval:metric_report_interval,

            stored_testing_data:stored_testing_data,
            performance_metrics:performance_metrics,
            normalization_statistics:normalization_statistics};

        const modelOut = {
            layers:{},
            properties:model_properties, 
            train(feature_data, target_data){
                train(this, feature_data, target_data)
            },
            test(){
                test(this)
            },
            set_features(){
                set_features(this)
            },
            set_features_csv(){
                set_features_csv(this)
            }};
        
        return modelOut;
    }
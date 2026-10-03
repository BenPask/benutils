//-------RANDOM-------
let seed = 23536;

export function zto_random(){
    const a = 49234
    const m = 1000000007 //large prime number

    //calc next seed
    seed = (seed*a)%m;

    //turn it into a number between 0 and 1
    return seed/m;
}

//-------MATRICES-------
export function transpose_matrix(m){

    let output = [];

    if(!Array.isArray(m[0])){
        if(m.length==1){
            return m
        }
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

    //return to 1-d if only 1 row
    if(output.length == 1){
        output = output[0]
    }
    
    return output;
}



export function matmult(m1, m2){
    
    //transpose m2 for easier multiplication
    m2 = transpose_matrix(m2)

    const mats = [m1,m2]
    //dealing with 1d array cases
    for(let i = 0; i<mats.length; i++){
        if(!Array.isArray(mats[i][0])){
            mats[i] = [mats[i]]
        }
    }

    [m1,m2] = mats;
    
    let m1_rows = m1.length;
    let m1_cols = m1[0].length;

    let m2_rows = m2.length;
    let m2_cols = m2[0].length;

    if(m1[0].length!=m2[0].length){
        throw new Error("The amount of columns in m1 must match the amount of rows in m2")
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
    
    if(product.length == 1){
        product = product[0]
    }
    return product;
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

//-------NEURAL_NETWORK-------
function generate_weight(){
    //b in the equation, constant variable to adjust variance of starting weights
    const sensitivity = 1

    const zeroToOne = zto_random();

    const weightOut = (zeroToOne*(2*sensitivity))-sensitivity;

    return weightOut;
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
    
    for(let i = 0;i<multiplied_output.length;i++){
        arr_out.push(multiplied_output[i]+bias_arr[i]);
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

function bwd_pass(model, layer_index){

}

function activate(model, arr_in, o_layer=false){
    if(o_layer == true){
        const acitvation_func = activation_map[model.properties.output_layer_info.activation];
        return acitvation_func(arr_in);
    }
    const acitvation_func = activation_map[model.properties.activation_type];
    return acitvation_func(arr_in);
}

function build_matrices(model){
    //build default structure CHANGE LATER
    model.layers["input_layer"] = {"output":[]};
    for(let i = 1;i<=model.properties.hidden_layer_info.count;i++){
        model.layers[`hidden_layer_${i}`] = {"main":{"weights":[],"biases":[]}, "delta":{"weights":[],"biases":[]},"preactivation":[], "output":[]};
    }
    model.layers["output_layer"] = {"main":{"weights":[],"biases":[]}, "delta":{"weights":[],"biases":[]}, "preactivation":[], "output":[]};

    //we have 1D arrays for the main & delta weights, we now add an amount of arrays(rows of matrix) for the size of the layer
    //This next line just lets us call the layers in order via index. 0 is the input layer
    const indexed_layers = Object.values(model.layers);

    //matrix builder, does not yet contain columns or weights ie: layers["hidden_layer_1"] = [[][][][][]] if hidden layer size was 5
    for(let i=1; i<indexed_layers.length-1; i++){
        for(let r = 0; r<model.properties.hidden_layer_info["init_size"];r++){
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
    let prev_layer_size = feature_count

    for(let i=1; i<indexed_layers.length; i++){
        for(const m_row of indexed_layers[i]["main"]["weights"]){
            //push a generated weight for the size of the previous layers output.
            for(let i = 0;i<prev_layer_size;i++) m_row.push(generate_weight());
        }
        //update the previous layer size variable to be this layer before moving on
        prev_layer_size = indexed_layers[i]["main"]["weights"].length

        for(let n = 0; n < indexed_layers[i]["main"]["weights"].length; n++){
            indexed_layers[i]["main"]["biases"].push(0);
        }
    }    
}

function train(model, feature_data, target_data){
    const observation_count = target_data["TARGET"].length
    //# of features being used NOT including target
    const feature_count = Object.keys(feature_data).length;
    
    //guardrail to confirm all features including target have the same number of observations
    for(const feat_name of Object.keys(feature_data)){
        if(observation_count != feature_data[feat_name].length){
            throw new Error("Features and targets are not all the same size!(or wrong input format, etc.)");
        }
    }



    //build simple structure without columns or weights, but including rows via the neuron count(size) of each layer
    build_matrices(model);
    fill_matrices(model, feature_count)
    iteration(model, 0, feature_data, target_data)
}

function iteration(model, obs_num, feature_data, target_data){
    model.layers["input_layer"]["output"] = [];
    
    const target_value  = target_data["TARGET"][obs_num]

    //set the model.layers["input_layer"]["output"] to the values of the feature's for this observation number
    for(const feat_name of Object.keys(feature_data)){
        model.layers["input_layer"]["output"].push(feature_data[feat_name][obs_num]);
    };

    for(let i = 1;i < Object.keys(model.layers).length; i++){
        fwd_pass(model, i)
    }

    //calculate error

}

function reluAcivation(arr){
    let arrOut = [];
    arrOut = arr.map(x => x > 0 ? x : 0);

    return arrOut
}

function derivReluActivation(arr){
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

function noActivation(arr){
    return arr;
}

function derivNoActivation(arr){
    return arr.map(x => 1)
}

const activation_map = {"none":noActivation,"relu":reluAcivation}
const deriv_acitvation_map = {"none":derivNoActivation,"relu":derivReluActivation}

export function neuralnet({
    //default parameters for neurat net initialization
    hidden_layer_info = {count:5, init_size:5, style:"decreasing"},
    activation_type = "relu",
    learn_rate = 0.01,
    output_layer_info = {init_size:1, activation:"none"},
    epochs = 100}={}) {

        console.log(`now buidling model, set for ${epochs} epochs, ${activation_type} activation, and a learning rate of ${learn_rate}`);

        const model_properties = {
            hidden_layer_info:hidden_layer_info,
            activation_type:activation_type,
            learn_rate:learn_rate,
            output_layer_info:output_layer_info,
            epochs:epochs};

        const modelOut = {
            layers:{},
            properties:model_properties, 
            train(feature_data, target_data){
                train(this, feature_data, target_data)
            }};
        
        return modelOut;
    }







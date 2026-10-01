//-------RANDOM-------
let seed = 82945;

export function zto_random(){
    const a = 49234
    const m = 5555555557 //large prime number

    //calc next seed
    seed = (seed*a)%m;

    //turn it into a number between 0 and 1
    return seed/m;
}

//-------MATRICES-------
export function transpose_matrix(m){

    const output = [];

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
    m2 = transpose_matrix(m2)

    //vars for # of rows
    const m1_rows = m1.length;
    const m2_rows = m2.length;

    //vars for # of columns
    const m1_cols = m1[0].length;
    const m2_cols = m2[0].length;

    if(m1_cols != m2_cols){
        throw new Error("m1 must have as many columns as m2 has rows")
    }

    const product = [];

    //for the # of rows in the first matrix
    for(let r_m1 = 0; r_m1<m1_rows; r_m1++){
        const cur_row_out = [];

        //for the # of rows in the transposition of m2(cols of m2)
        for(let r_m2 = 0; r_m2<m2_rows; r_m2++){
            let cur_value_out = 0;

            //for the # of columns in the first matrix
            for(let c_m1 = 0; c_m1<m1_cols; c_m1++){
                cur_value_out += m1[r_m1][c_m1]*m2[r_m2][c_m1];
            }
            cur_row_out.push(cur_value_out);
        }
        product.push(cur_row_out)
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
    const sensitivity = 0.1

    const zeroToOne = zto_random();

    const weightOut = (zeroToOne*(2*sensitivity))-sensitivity;

    return weightOut;
}

function fwd_pass(layer_index){
    const i_layers = Object.values(this.layers);

    const prev_layer_output = i_layers[layer_index-1]["output"];
    const weight_matrix = i_layers[layer_index]["weights"];

    i_layers[layer_index]["output"] = matmult(weight_matrix, prev_layer_output);
    
}

function set_import_values(){
    
}

function iteration(feature_data, target_data, i){

    const i_layers = Object.values(this.layers);

    const in_layer_arr = [];

    //push the observations input values into the input_layer's "output":x
    for(const feat_name of Object.keys(feature_data)){
        in_layer_arr.push(feature_data[feat_name][i]);
        this.layers["input_layer"]["output"] = in_layer_arr;
    }

    console.log(i_layers);
}

function build_matrices(feature_data){
    const in_layer_size = Object.keys(feature_data).length; 

    //build default structure CHANGE LATER
    this.layers["input_layer"] = {"output":[]};
    for(let i = 1;i<=this.properties.hidden_layer_info.count;i++){
        this.layers[`hidden_layer_${i}`] = {"weights":[], "delta":[], "output":[]};
    }
    this.layers["output_layer"] = {"weights":[], "delta":[], "output":[]};


    //Start generating weights and layer matrices
    const indexed_layers = Object.values(this.layers);

    let prev_layer_ncount = in_layer_size;

    //weight generation and matrix builder
    for(let i=1; i<indexed_layers.length; i++){
        for(let r = 0; r<this.properties.hidden_layer_info["init_size"];r++){
            const row_out = []
            for(let c = 0; c<prev_layer_ncount; c++){
                row_out.push(generate_weight())
            }
            indexed_layers[i]["weights"].push(row_out);
        }
        prev_layer_ncount = indexed_layers[i]["weights"].length
    }    
}

function train(feature_data, target_data){
    const observation_count = target_data["TARGET"].length
    
    
    for(const feat_name of Object.keys(feature_data)){
        if(observation_count != feature_data[feat_name].length){
            throw new Error("Features and targets are not all the same size!(possibly format)");
        }
    }

    this.build_matrices(feature_data);
    
    //for(const e = 0;e<this.epochs;e++){
        for(let i = 0;i<observation_count;i++){
            console.log(this.iteration(feature_data, target_data, i))
        }
    //}
}

export function neuralnet({
    //default parameters for neurat net initialization
    hidden_layer_info = {count:5, init_size:5, style:"decreasing"},
    activation_type = "relu",
    learn_rate = 0.01,
    epochs = 100}={}) {

        console.log(`now buidling model, set for ${epochs} epochs, ${activation_type} activation, and a learning rate of ${learn_rate}`);

        const model_properties = {
            hidden_layer_info:hidden_layer_info,
            activation_type:activation_type,
            learn_rate:learn_rate,
            epochs:epochs};


        const modelOut = {layers:{},properties:model_properties, train};
        
        return modelOut;
    }







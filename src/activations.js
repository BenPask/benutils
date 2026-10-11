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

export function activate(model, arr_in, o_layer=false){
    if(o_layer == true){
        const acitvation_func = activation_map[model.properties.output_layer_info.activation];
        return acitvation_func(arr_in);
    }
    const acitvation_func = activation_map[model.properties.activation_type];
    return acitvation_func(arr_in);
}

export function deriv_activate(model, arr_in, o_layer=false){
    if(o_layer == true){
        const acitvation_func = deriv_acitvation_map[model.properties.output_layer_info.activation];
        return acitvation_func(arr_in);
    }
    const acitvation_func = deriv_acitvation_map[model.properties.activation_type];
    return acitvation_func(arr_in);
}
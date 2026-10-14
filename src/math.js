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

    if(power == 0 || value == 1){
        return 1;
    }

    if(value == 0){
        return 0;
    }



    let is_negative = power < 0;
    let effective_power = is_negative ? -power : power;

    let current = 1;

    if(effective_power < 1){
        return root(value, 1 / effective_power);
    } else if(Number.isInteger(effective_power)){
        for(let i = 0; i<effective_power-1; i++){
        current = current*value;
        }
    } else {
        let whole_num = round_float(effective_power-0.5);
        let partial_num = effective_power - whole_num;

        let whole_result = value;
        for (let i = 0; i < whole_num - 1; i++){
            whole_result = whole_result * value;
        }

        let partial_result = root(value, 1/partial_num);

        current = whole_result * partial_result;
    }
    
    if(is_negative){
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

export function root(value, nth_root, tolerance=1e-15){
    if(value<0){
        throw new Error("Cannot take the square root of negative numbers")
    }
    if(value == 0){
        return 0;
    }

    let guess = 0;

    if(value < 1){
        guess = 1;
    } else {
        guess = value/nth_root;
    }
    
    let iter = 0;

    const divisor = 1/nth_root;

    while(iter < 1000){
        
        let denom = xpo(guess, nth_root - 1);
        let next_guess = divisor*((nth_root-1) * guess + value / denom);

        if(absolute_value(guess - next_guess) < tolerance){
            return next_guess
        }

        guess = next_guess;
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
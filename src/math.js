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
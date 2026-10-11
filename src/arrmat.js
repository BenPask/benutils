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

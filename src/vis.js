export function scatter({
    x_values, 
    y_values,
    width, 
    height, 
    x_label = "x-axis", 
    y_label = "y-axis", 
    x_step = null, 
    y_step = null, 
    grid_lines = true,
    overdraw = 0,
    connect_points = false,
    class_name = "graph",
    background_color = "lightBlue",
    point_size = 2,
}){
    if(x_values.length != y_values.length){
        throw new Error("the lenth of array x does not equal the length of array y");
    }

    const x = {
        values:x_values,
        max:x_values[0],
        min:x_values[0]
    }

    const y = {
        values:y_values,
        max:y_values[0],
        min:y_values[0]
    }

    const points = [];
    
    for(let i = 0; i < x_values.length; i++){
        x.max = x_values[i] > x.max ? x_values[i] : x.max;
        x.min = x_values[i] < x.min ? x_values[i] : x.min;
        points.push({x:x_values[i], y:null})
    }
    
    for(let i = 0; i < y_values.length; i++){
        y.max = y_values[i] > y.max ? y_values[i] : y.max;
        y.min = y_values[i] < y.min ? y_values[i] : y.min;
        points[i].y = y_values[i];
    }
    
    x_step = x_step == null ?  x.max/(x_values.length-1) : x_step;
    y_step = y_step == null ?  y.max/(y_values.length-1) : y_step;

    const axis_point_mod = 8
    

    const starting_axis_point = {x:width/axis_point_mod, y:height/axis_point_mod-axis_point_mod/2};
    const origin = {x:starting_axis_point.x, y:(starting_axis_point.y*(axis_point_mod -1))};
    const finishing_axis_point = {x:(origin.x)*(axis_point_mod -1), y:origin.y};

    const graph = {
        x:x,
        y:y,

        x_label:x_label,
        y_label:y_label,

        x_step:x_step,
        y_step:y_step,

        width:width,
        height:height,

        grid_lines:grid_lines,
        overdraw:overdraw,
        point_size:point_size,

        connect_points:connect_points,

        starting_axis_point:starting_axis_point,
        origin:origin,
        finishing_axis_point:finishing_axis_point,

        background_color:background_color,
        class_name:class_name,

        points:points
    }

    graph.plot = function(){
        return plot(this);
    };
    
    return graph
}

function draw_axes(ctx, graph){
    ctx.beginPath();
    ctx.moveTo(graph.starting_axis_point.x, graph.starting_axis_point.y);
    ctx.lineTo(graph.origin.x, graph.origin.y);
    ctx.lineTo(graph.finishing_axis_point.x, graph.finishing_axis_point.y);
    ctx.stroke();
}

function draw_grid_lines(ctx, graph){
    const delta_y = -1 * (y_to_px(graph, graph.y_step) - graph.origin.y).toFixed(3)
    const delta_x = -1 * (x_to_px(graph, graph.x_step) - graph.origin.x).toFixed(3)


    const cur_point = {x:graph.finishing_axis_point.x, y:graph.starting_axis_point.y};

    for(let i = 0; i < graph.y.max/graph.y_step; i++){
        ctx.beginPath();
        ctx.strokeStyle = "grey";
        ctx.moveTo(graph.starting_axis_point.x - graph.overdraw, cur_point.y);
        ctx.lineTo(graph.finishing_axis_point.x, cur_point.y);
        ctx.stroke();
        ctx.textAlign = 'right'
        ctx.fillText(`${((graph.y.max/graph.y_step-i)*graph.y_step).toFixed(3)}`, graph.starting_axis_point.x - graph.overdraw - 3, cur_point.y + 3)
        
        cur_point.y += delta_y;
    }
    for(let i = 0; i < graph.x.max/graph.x_step; i++){
        
        ctx.beginPath();
        ctx.strokeStyle = "grey";
        ctx.moveTo(cur_point.x, graph.starting_axis_point.y);
        ctx.lineTo(cur_point.x, graph.finishing_axis_point.y + graph.overdraw);
        ctx.stroke();
        ctx.textAlign = 'center'

        ctx.restore();
        ctx.save();
        ctx.translate( cur_point.x+3 , graph.finishing_axis_point.y + graph.overdraw + 18)
        ctx.rotate(45 * Math.PI / 180)
        ctx.fillText(`${((graph.x.max/graph.x_step-i)*graph.x_step).toFixed(3)}`, 0 , 0)
        ctx.restore();
  
        cur_point.x += delta_x;
    }
}

function x_to_px(graph, x_in){
    return ((x_in/graph.x.max)*(graph.finishing_axis_point.x - graph.origin.x) + graph.origin.x);
}

function y_to_px(graph, y_in){
    return (-1 * (y_in/graph.y.max)*(graph.origin.y - graph.starting_axis_point.y) + graph.origin.y);
}


function draw_points(ctx, graph){

    for(const point of graph.points){
        ctx.beginPath();
        ctx.strokeStyle = "black";
        ctx.arc(x_to_px(graph, point.x), y_to_px(graph, point.y), graph.point_size, 0, Math.PI*2);
        ctx.fill();
    }
}

function draw_points_line(ctx, graph){
        ctx.beginPath();
        ctx.moveTo(x_to_px(graph, graph.points[0].x), y_to_px(graph, graph.points[0].y));
        for(let i = 1; i < graph.points.length; i++){
            ctx.lineTo(x_to_px(graph, graph.points[i].x), y_to_px(graph, graph.points[i].y));
        }
        ctx.stroke();
    }


export function plot(graph){
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext("2d");

    canvas.setAttribute('class', `${graph.class_name}`);

    canvas.width = graph.width;
    canvas.height = graph.height;
    
    draw_axes(ctx, graph);

    if(graph.grid_lines){
        draw_grid_lines(ctx, graph);
    }

    draw_points(ctx, graph);

    if(graph.connect_points){
        draw_points_line(ctx, graph);
    }
    

    ctx.textAlign = 'center'
    ctx.font = '20px serif'
    ctx.fillText(`${graph.x_label}`, graph.width/2, graph.height*.93)



    ctx.save();
    ctx.translate(graph.width*.06, graph.height/2)
    ctx.rotate(270 * Math.PI / 180)
    
    ctx.font = '20px serif'
    ctx.fillText(`${graph.y_label}`, 0 , 0)
    ctx.restore();
  
    
    
    canvas.setAttribute('style', `background-color: ${graph.background_color}; width: ${graph.width}px; height: ${graph.height}px;`);

    return canvas;
}



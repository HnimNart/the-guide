use crate::gpu_vector::GPUVector;
use crate::utility::{
    are_vectors_equivalent_mse_scale, mean_square_error, measure_time, run_compute_shader,
    GPUHandles, Uniform, compare_vectors,
};

fn matrix_multiplication_cpu(
    left_matrix: &Vec<f32>,
    right_matrix: &Vec<f32>,
    outer_dimension_left_length: usize,
    inner_dimension_length: usize,
    outer_dimension_right_length: usize,
) -> Vec<f32> {
    let mut output: Vec<f32> =
        vec![0.0; outer_dimension_left_length * outer_dimension_right_length];
    for row_output in 0..outer_dimension_left_length {
        for column_output in 0..outer_dimension_right_length {
            for inner_dimension in 0..inner_dimension_length {
                output[row_output * outer_dimension_right_length + column_output] += left_matrix
                    [row_output * inner_dimension_length + inner_dimension]
                    * right_matrix[inner_dimension * outer_dimension_right_length + column_output];
            }
        }
    }
    output
}

fn run_matmul_shader_tiled(
    handles: &GPUHandles,
    shader: &'static str,
    shader_func: &str,
    m: usize,
    n: usize,
    tiled_size:usize,
    uniform: &Uniform,
    mat1: &GPUVector,
    mat2: &GPUVector,
    verbose: bool,
) -> Vec<f32> {
    let output: Vec<f32> = vec![0.0; m * n];
    let mut output_gpu: GPUVector = GPUVector::new(&handles, output, "output", true);

    let block_size: usize = tiled_size;
    let launch_blocks_x: u32 = ((m + block_size - 1) / block_size) as u32;
    let launch_blocks_y: u32 = ((n + block_size - 1) / block_size) as u32;

    if verbose {
        println!("Dispatching {} x blocks of {} threads and {} y blocks of {} threads each for a total of {} threads!", launch_blocks_x, block_size, launch_blocks_y, block_size, launch_blocks_x as usize * launch_blocks_y as usize * block_size * block_size);
    }

    // Reuse this function for the convolution and matrix multiplication tasks
    run_compute_shader(
        handles,
        block_size,
        launch_blocks_x,
        block_size,
        launch_blocks_y,
        shader,
        shader_func,
        &uniform,
        &mat1,
        &mat2,
        &mut output_gpu,
    );
    return output_gpu.cpu_data;
}

fn run_matmul_shader(
    handles: &GPUHandles,
    shader: &'static str,
    shader_func: &str,
    m: usize,
    n: usize,
    uniform: &Uniform,
    mat1: &GPUVector,
    mat2: &GPUVector,
    verbose: bool,
) -> Vec<f32> {
    let output: Vec<f32> = vec![0.0; m * n];
    let mut output_gpu: GPUVector = GPUVector::new(&handles, output, "output", true);

    let block_size_x: usize = 32;
    let launch_blocks_x: u32 = (((m * n) + block_size_x - 1) / block_size_x) as u32;
    let block_size_y: usize = 1;
    let launch_blocks_y: u32 = 1;

    if verbose {
        println!("Dispatching {} x blocks of {} threads and {} y blocks of {} threads each for a total of {} threads!", launch_blocks_x, block_size_x, launch_blocks_y, block_size_y, launch_blocks_x as usize * launch_blocks_y as usize * block_size_x * block_size_y);
    }

    // Reuse this function for the convolution and matrix multiplication tasks
    run_compute_shader(
        handles,
        block_size_x,
        launch_blocks_x,
        block_size_y,
        launch_blocks_y,
        shader,
        shader_func,
        &uniform,
        &mat1,
        &mat2,
        &mut output_gpu,
    );
    return output_gpu.cpu_data;
}

fn test_ground_truth() -> bool {
    let outer_dimension_left: usize = 4;
    let inner_dimension: usize = 3;
    let outer_dimension_right: usize = 3;
    let left: Vec<f32> = vec![1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0];
    let right: Vec<f32> = vec![1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 2.0];
    let ground_truth_output: Vec<f32> =
        vec![5.0, 4.0, 3.0, 8.0, 9.0, 5.0, 6.0, 5.0, 3.0, 11.0, 9.0, 6.0];

    let output: Vec<f32> = matrix_multiplication_cpu(
        &left,
        &right,
        outer_dimension_left,
        inner_dimension,
        outer_dimension_right,
    );

    assert!(output.len() == ground_truth_output.len());
    for index in 0..ground_truth_output.len() {
        if 0.00001 < (output[index] - ground_truth_output[index]).abs() {
            return false;
        }
    }

    true
}

pub fn matrix_multiplication(handles: &GPUHandles) -> bool {
    // A small test to ensure that the matrix_multiplication_cpu function is actually correct.
    let ground_truth_is_correct: bool = test_ground_truth();
    println!(
        "Matrix multiplication ground truth function is correct: {}",
        ground_truth_is_correct
    );
    assert!(ground_truth_is_correct);
    let number_of_runs = 10;
    let tiled_size = 8;

    // Use big data dimensions to make sure the cost of transferring
    // doesn't dominate the time spent in the function.
    let outer_dimension_left: usize = 32; // M
    let inner_dimension: usize = 32; // N
    let outer_dimension_right: usize = 32; // K
    let scale = 0.1;
    let left_matrix: Vec<f32> = (0..outer_dimension_left * inner_dimension)
        .map(|x| x as f32 * scale)
        .collect();
    let right_matrix: Vec<f32> = (0..inner_dimension * outer_dimension_right)
        .map(|x| x as f32 * scale)
        .collect();
    let ground_truth: Vec<f32> = matrix_multiplication_cpu(
        &left_matrix,
        &right_matrix,
        outer_dimension_left,
        inner_dimension,
        outer_dimension_right,
    );

    //
    // 1) Make one version of matrix multiplication using the GPU. Ensure that it is correct.
    //
    // 2) Make another version which uses tiling through shared memory and local accumulation in a register.
    // A tiling reference: http://www.csce.uark.edu/~mqhuang/courses/4643/s2016/lecture/GPU_Lecture_3.pdf
    //
    // 3) After ensuring correctness - time the two functions.
    //
    // 4) How big do the matrices have to be before you see a big performance difference?
    //
    // 5) What happens when you set the block size to different multiples of 32? Why do you think that is?
    //
    // 6) What is the optimal tile size?
    //
    // 7) Make a third version starting from the tiled version, but pads the matrices with 0's to the nearest
    // multiple of the block size? So, if you had a block size of 32 and a 30x30 * 28x29
    // multiplication you padded both with 0's to get 32x32 * 32x32.
    // HINT - You can now remove some if-guards.
    //
    // HINT - You need a run_compute_shader() call per type of compute shader.
    // Figure out what the arguments are supposed to be (see vector_add.rs) and
    // call the correct shader function in the correct shader file.
    //

    // UNIFORM data
    let uniform: Uniform = Uniform::new(
        handles,
        outer_dimension_left,
        inner_dimension,
        outer_dimension_right,
        tiled_size,
    );

    // Input data
    let mat1: GPUVector = GPUVector::new(&handles, left_matrix, "mat1", false);
    let mat2: GPUVector = GPUVector::new(&handles, right_matrix, "mat2", false);

    //
    // YOUR CODE HERE
    let data_naive: Vec<f32> = measure_time("Naive", number_of_runs, |verbose: bool| {
        run_matmul_shader(
            handles,
            include_str!("matrix_multiplication_naive.wgsl"),
            "matmul_naive",
            outer_dimension_left,
            outer_dimension_right,
            &uniform,
            &mat1,
            &mat2,
            verbose,
        )
    });
    let data_tiled: Vec<f32> = measure_time("Tiled", number_of_runs, |verbose: bool| {
        run_matmul_shader_tiled(
            handles,
            include_str!("matrix_multiplication_tiled.wgsl"),
            "matmul_tiled",
            outer_dimension_left,
            outer_dimension_right,
            tiled_size,
            &uniform,
            &mat1,
            &mat2,
            verbose,
        )
    });

    // let data_padded: Vec<f32> = ground_truth.clone(); // Remove this and replace with your own data

    // Naive
    println!(
        "matrix multiplication naive MSE: {}",
        mean_square_error(&ground_truth, &data_naive)
    );
    let success: bool = are_vectors_equivalent_mse_scale(
        &ground_truth,
        &data_naive,
        (inner_dimension * inner_dimension) as f64,
    );
    println!("matrix multiplication naive success: {}!", success);

    // Tiled
    println!(
        "matrix multiplication tiled MSE: {}",
        mean_square_error(&ground_truth, &data_tiled)
    );
    let success: bool = compare_vectors(
        &ground_truth,
        &data_tiled
    );
    println!("matrix multiplication tiled success: {}!", success);

    // // Padded
    // println!(
    //     "matrix multiplication padded MSE: {}",
    //     mean_square_error(&ground_truth, &data_padded)
    // );
    // let success: bool = are_vectors_equivalent(&ground_truth, &data_padded);
    // println!("matrix multiplication padded success: {}!", success);

    true
}

struct Uniform {
    element_count: u32,
    filter_size: u32,
    filter_size_2: u32,
    not_used: u32,
};

@group(0) @binding(0)
var<uniform> dims: Uniform;

// Bind a read only array of 32-bit floats
@group(0) @binding(1)
var<storage, read> input_signal: array<f32>;

@group(0) @binding(2)
var<storage, read> input_filter: array<f32>;

// Bind a read/write array
@group(0) @binding(3)
var<storage, read_write> output: array<f32>;


@compute @workgroup_size(32, 1, 1) 

fn convolution_naive(

    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Signal index start
    let thread_id: u32 = global_id.x;

    // Just to be sure
    if (thread_id < dims.element_count) {
        output[thread_id] = 0.0f;
    }

    var i : u32  = u32(0);
    // i is index into filter
    while (i < dims.filter_size) {
        let offset:  u32= thread_id - dims.filter_size_2 + u32(i);
        if (u32(0) <= offset && offset < dims.element_count) {
            output[thread_id] += input_signal[offset] * input_filter[i];
        }
        i = i + u32(1);
    }
}
struct Uniform {
    element_count : u32,
    filter_size : u32,
    filter_size_2 : u32,
    not_used : u32,
};

@group(0) @binding(0)
var<uniform> dims : Uniform;

//Bind a read only array of 32-bit floats
@group(0) @binding(1)
var<storage, read> input_signal : array<f32>;

@group(0) @binding(2)
var<storage, read> input_filter : array<f32>;

//Bind a read/write array
@group(0) @binding(3)
var<storage, read_write> output : array<f32>;

const SIGNAL_SIZE : u32 = 50u; 
//var<workgroup> workgroup_filter: array<f32, workgroup_len>;
var<workgroup> workgroup_signal : array<f32, SIGNAL_SIZE>;


@compute @workgroup_size(32, 1, 1)
fn convolution_shared(
@builtin(local_invocation_id) local_id : vec3 < u32>,
@builtin(global_invocation_id) global_id : vec3 < u32>,
)
{
    //Signal index start
    let thread_id : u32 = global_id.x;

    //Just to be sure
    if (thread_id < dims.element_count)
    {
        output[thread_id] = 0.0f;
    }

    if (local_id.x == u32(0))
    {
        var j : u32 = u32(0);
        while (j < u32(SIGNAL_SIZE))
        {
            let offset = thread_id - dims.filter_size_2 + u32(j);

            if (offset >= u32(0) && offset < dims.element_count)
            {
                workgroup_signal[j] = input_signal[offset];
            } else {
                workgroup_signal[j] = f32(0);
            }
            j = j + u32(1);
        }
    }

    // Load filter into shared data
    //workgroup_filter[local_id.x] = input_filter[local_id.x];
    workgroupBarrier();

    var i : u32 = u32(0);
    while (i < dims.filter_size)
    {
        output[thread_id] += workgroup_signal[local_id.x + i] * input_filter[i];
        i = i + u32(1);
    }

    return;
}

struct Uniform {
    m : u32,
    n : u32,
    p : u32,
    not_used : u32
};

@group(0) @binding(0)
var<uniform> dims : Uniform;

//Bind a read only array of 32-bit floats
@group(0) @binding(1)
var<storage, read> mat1 : array<f32>;

@group(0) @binding(2)
var<storage, read> mat2 : array<f32>;

//Bind a read/write array
@group(0) @binding(3)
var<storage, read_write> output : array<f32>;


@compute @workgroup_size(32, 1, 1)

fn matmul_naive(

@builtin(global_invocation_id) global_id : vec3 < u32>,
)
{
    //Signal index start
    let thread_id : u32 = global_id.x;
    let m : u32 = dims.m;
    let n : u32 = dims.n;
    let p : u32 = dims.p;


    //get row
    let i = thread_id / p;
    //get column
    let j = thread_id % p;

    if (i * p + j > (m * p))
    {
        return;
    }

    for (var k = u32(0); k < n; k = k + u32(1))
    {
        output[i * p + j] += mat1[i * n+ k] * mat2[k * p+ j];
    }


}

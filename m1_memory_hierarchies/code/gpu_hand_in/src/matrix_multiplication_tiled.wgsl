struct Uniform {
    m: u32,
    n: u32,
    p: u32,
    tiled_size: u32
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


const block_size: u32 = 8u;
const BLOCK_SIZE_2D: u32 = 64u;
var<workgroup> Mds : array<f32, BLOCK_SIZE_2D>;
var<workgroup> Nds : array<f32, BLOCK_SIZE_2D>;

@compute @workgroup_size(8u, 8u, 1u)

fn matmul_tiled(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    //Signal index start
    let m: u32 = dims.m;
    let n: u32 = dims.n;
    let p: u32 = dims.p;

    //get row
    // let i = global_id.x / p;
    let i = workgroup_id.y * block_size + local_id.y;
    //get column
    // let j = global_id.x % p;
    let j = workgroup_id.x * block_size + local_id.x;


    var value = 0.0;
    for (var m = 0u; m < u32(p / block_size); m++) {
        Mds[local_id.y * block_size + local_id.x] = mat1[i * n + (m * block_size + local_id.x)];
        Nds[local_id.y * block_size + local_id.x] = mat2[(m * block_size + local_id.y) * p + j];
        workgroupBarrier();

        for (var k = 0u; k < block_size; k++) {
            value += Mds[local_id.y * block_size + k] * Nds[k * block_size + local_id.x];
        }
        workgroupBarrier();
    }
    output[i * p + j] = value;
}

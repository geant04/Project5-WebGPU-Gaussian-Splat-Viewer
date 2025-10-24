const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    packedPositionOpacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    radius: f32,
    ndc_position: vec2f,
    conicAndOpacity: vec4f
};

//TODO: bind your data here
@group(0) @binding(0) var<storage, read> gaussians : array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> splats : array<Splat>;

@group(1) @binding(0) var<storage, read_write> sort_infos: SortInfos;
@group(1) @binding(1) var<storage, read_write> sort_depths : array<u32>;
@group(1) @binding(2) var<storage, read_write> sort_indices : array<u32>;
@group(1) @binding(3) var<storage, read_write> sort_dispatch: DispatchIndirect;

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    return vec3<f32>(0.0);
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

fn computeGaussianBBIntersection(position: vec3f, viewMatrix: mat4x4f, projMatrix: mat4x4f, pView: vec3f) -> bool {
    // points to screenspace
    return true;
}


fn quaternionToRotationMatrix(quaternion: vec4f) -> mat3x3f {
    // I don't like this, but it's the style of the Inria paper code
    let r = quaternion.x;
    let x = quaternion.y;
    let y = quaternion.z;
    let z = quaternion.w;

    let column0 = vec3f(1f - 2f * (y * y + z * z), 2f * (x * y - r * z), 2f * (x * z + r * y));
    let column1 = vec3f(2f * (x * y + r * z), 1f - 2f * (x * x + z * z), 2f * (y * z - r * x));
    let column2 = vec3f(2f * (x * z - r * y), 2f * (y * z + r * x), 1f - 2f * (x * x + y * y));

    return mat3x3f(
        column0,
        column1,
        column2
    );
}

fn buildScaleMatrix(scale: vec4f) -> mat3x3f {
    let scaleMatrix = glm::mat3x3f(1f);

    // somehow grab gaussian uniforms when it's populated
    let column0 = vec3f(scale.x, 0f, 0f);
    let column1 = vec3f(0f, scale.y, 0f);
    let column2 = vec3f(0f, 0f, scale.z);
    return mat3x3f(column0, column1, column2);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;

    // TODO: set up pipeline as described in instruction
    // Somehow... we do frustum culling
    let gaussian = gaussians[idx];

    let xy = unpack2x16float(gaussian.packedPositionOpacity[0]);
    let za = unpack2x16float(gaussian.packedPositionOpacity[1]);
    let worldPos = vec4<f32>(xy[0], xy[1], za[0], 1.);

    // Project gaussian world position to NDC?
    let projPosW : vec4f = proj * view * vec4f(worldPos);
    let projPos : vec4f = projPosW / projPosW.w;

    // Compute covariance 3D matrix
    let quaternion : vec4f = (unpack2x16float(gaussian.rot[0]), unpack2x16float(gaussian.rot[1]));
    let scale : vec4f = (unpack2x16float(gaussian.scale[0], unpack2x16float(gaussian.scale[1])));

    let rotMat = quaternionToRotationMatrix(quaternion);
    let scaleMat = buildScaleMatrix(scale);

    let m : mat3x3f = scaleMat * rotMat;
    let sigma : mat3x3f = transpose(m) * m;

    let cov3D : mat3x3f = 


    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys

    // dude what the hell is dispatchIndirect and why does the base code just not talk about it at all??
}
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    radius: f32,
    ndc_position: vec2f,
    color: vec4f,
    conicAndOpacity: vec4f
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0) var<storage, read> splats : array<Splat>;
@group(0) @binding(1) var<uniform> camera: CameraUniforms;

@vertex
fn vs_main( @builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32 ) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    const pos = array(
        vec2f(-1f, -1f), vec2f(1f, -1f), vec2f(-1f, 1f),
        vec2f(-1f, 1f), vec2f(1f, -1f), vec2f(1f, 1f)
    );

    const smallRadius = 0.01f;
    let splatPosition: vec2f = splats[instanceIndex].ndc_position;

    out.position = vec4f(pos[vertexIndex] * smallRadius + splatPosition, 0f, 1f);

    //out.position = vec4<f32>(1. ,1. , 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1f, 1f, 0f, 0f);
}
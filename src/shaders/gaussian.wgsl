struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4f,
    @location(1) conic: vec3f,
    @location(2) splatCenter: vec2f
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
@group(0) @binding(2) var<storage, read> sort_indices : array<u32>;

@vertex
fn vs_main( @builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32 ) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 

    let sortedIdx = sort_indices[instanceIndex];
    var out: VertexOutput;

    const pos = array(
        vec2f(-1f, -1f), vec2f(1f, -1f), vec2f(-1f, 1f),
        vec2f(-1f, 1f), vec2f(1f, -1f), vec2f(1f, 1f)
    );

    let splatPixelRadius : f32 = (splats[sortedIdx].radius);
    let splatSize : vec2f = vec2f(splatPixelRadius, splatPixelRadius) / camera.viewport;
    let splatPosition: vec2f = splats[sortedIdx].ndc_position;
    let splatConicAndOpacity = splats[sortedIdx].conicAndOpacity;
    let splatColor = splats[sortedIdx].color.xyz;

    out.position = vec4f(pos[vertexIndex] * splatSize + splatPosition, 0f, 1f);
    out.color = vec4f(splatColor, splatConicAndOpacity.w);

    // Flip -1 for y, since [-1,1] needs to map to [h, 0]
    out.splatCenter = (splatPosition * vec2f(0.5f, -0.5f) + 0.5f) * camera.viewport;
    out.conic = splatConicAndOpacity.xyz;

    //out.position = vec4<f32>(1. ,1. , 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let splatPixelPosition : vec2f = in.splatCenter;
    let pixelPosition : vec2f = in.position.xy;
    let conic : vec3f = in.conic;

    let d = pixelPosition - splatPixelPosition;
    let power : f32 = -0.5f * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0f)
    {
        discard;
    }

    let alpha = min(0.99f, in.color.a * exp(power));
    if (alpha < (1f / 255f))
    {
        discard;
    }
    
    // The INRIA paper itself blends based on adding alphas together to accumulate an overall final pixel color 
    return vec4f(in.color.xyz * alpha, alpha);
}
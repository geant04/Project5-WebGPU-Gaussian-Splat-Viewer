struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    // position
    // scale
    // rotation
    // color
    // opacity
    radius: f32,
    ndc_position: vec2f,
    conicAndOpacity: vec4f
};

@vertex
fn vs_main( @builtin(vertex_index) VertexIndex: u32 ) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    const pos = array(
        vec2f(-1f, -1f), vec2f(1f, -1f), vec2f(-1f, 1f),
        vec2f(-1f, 1f), vec2f(1f, -1f), vec2f(1f, 1f)
    );

    const smallRadius = 0.01f;

    out.position = vec4f(pos[VertexIndex] * smallRadius , 0f, 1f);

    //out.position = vec4<f32>(1. ,1. , 0., 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1f, 1f, 0f, 0f);
}
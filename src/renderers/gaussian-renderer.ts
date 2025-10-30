import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import gaussianWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {

}

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
): GaussianRenderer {
  const testPreProcess = true;
  const sorter = get_sorter(pc.num_points, device);
  
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================
  const nulling_data = new Uint32Array([0]);

  const numSplats = pc.num_points;
  const floatsPerSplat = 4 + 4 + 4;
  const splats = new Float32Array(numSplats * floatsPerSplat);
  const splatStorageBuffer = device.createBuffer({
    label: "splats",
    size: splats.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(splatStorageBuffer, 0, splats);
  console.log("created a buffer for " + pc.num_points + " splats");

  const indirectDrawArgs = new Uint32Array([6, numSplats, 0, 0]);
  const indirectDrawBuffer = device.createBuffer({
    label: "indirect draw buffer stats",
    size: indirectDrawArgs.byteLength,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(indirectDrawBuffer, 0, indirectDrawArgs);

  const renderSettingsBuffer = device.createBuffer({
    label: "renderSettings buffer",
    size: 4 + 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(renderSettingsBuffer, 0, new Float32Array([1, pc.sh_deg]));

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const gaussian_shader = device.createShaderModule({code: gaussianWGSL });
  let preprocessPipeline;
  let gaussianBindGroup;
  let sortBindGroupLayout;
  let sortBindGroup;

  if (testPreProcess)
  {
    const gaussianBindGroupLayout = device.createBindGroupLayout({
      label: "gaussian bind group layout, containing gaussian and splat buffers",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" }
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage"}
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" }
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" }
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" }
        }
      ]
    });

    gaussianBindGroup = device.createBindGroup({
      label: 'gaussian bind group',
      layout: gaussianBindGroupLayout,
      entries: [
        { 
          binding: 0,
          resource: { buffer: pc.gaussian_3d_buffer }
        },
        {
          binding: 1,
          resource: { buffer: splatStorageBuffer }
        },
        {
          binding: 2,
          resource: { buffer: camera_buffer }
        },
        {
          binding: 3,
          resource: { buffer: pc.sh_buffer }
        },
        {
          binding: 4,
          resource: { buffer: renderSettingsBuffer }
        }
      ]
    });

    sortBindGroupLayout = device.createBindGroupLayout({
      label: "sort bind group layout",
      // there is apparently some need for all of these buffers to be read/write that I don't understand yet
      entries: [
        { 
          binding: 0, 
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        { 
          binding: 1, 
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        { 
          binding: 2, 
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        },
        { 
          binding: 3, 
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" }
        }
      ]
    });

    sortBindGroup = device.createBindGroup({
      label: 'sort bind group',
      layout: sortBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
        { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      ],
    });

    preprocessPipeline = device.createComputePipeline({
      label: 'preprocess',
      layout: device.createPipelineLayout({
        bindGroupLayouts: [
          gaussianBindGroupLayout,
          sortBindGroupLayout
        ]
      }),
      compute: {
        module: device.createShaderModule({ code: preprocessWGSL }),
        entryPoint: 'preprocess',
        constants: {
          workgroupSize: C.histogram_wg_size,
          sortKeyPerThread: c_histogram_block_rows,
        },
      },
    });
  }


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================

  const splatDrawBindGroupLayout = device.createBindGroupLayout({
    label: "splat draw bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "read-only-storage" }
      }
    ]
  });

  const splatDrawBindGroup = device.createBindGroup({
    label: "draw splat bind group",
    layout: splatDrawBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: splatStorageBuffer}
      },
      {
        binding: 1,
        resource: { buffer: camera_buffer}
      },
      {
        binding: 2,
        resource: { buffer: sorter.ping_pong[0].sort_indices_buffer }
      }
    ]
  });

  const splatPipeline = device.createRenderPipeline({
    label: 'render',
    layout: device.createPipelineLayout({
      bindGroupLayouts: [splatDrawBindGroupLayout]
    }),
    vertex: {
      module: gaussian_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: gaussian_shader,
      entryPoint: 'fs_main',
      targets: [
        { 
          format: presentation_format,
          blend: {
            color: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add'
            },
            alpha: {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha',
              operation: 'add'
            }
          }
        }
      ],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    // After this point, and after culling, we should have our num points properly found and updated?
    const pass = encoder.beginRenderPass({
      label: 'gaussian quad renderer',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    
    pass.setPipeline(splatPipeline);
    pass.setBindGroup(0, splatDrawBindGroup);
    pass.drawIndirect(indirectDrawBuffer, 0);

    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // Wipe out buffers, but only the first element I think
      // In this case:
      //  sort_info_buffer -> clear key_size, which is the first element (4 bytes, 0 offset)
      //  sort_dispatch_indirect_buffer -> clear dispatch_x, which is also the first element
      encoder.clearBuffer(sorter.sort_info_buffer, 0, 4);
      encoder.clearBuffer(sorter.sort_dispatch_indirect_buffer, 0, 4);

      // Need to dispatch the compute pass.
      if (testPreProcess)
      {
        const computePass = encoder.beginComputePass({
          label: 'pre-process compute pass'
        });

        computePass.setPipeline(preprocessPipeline);
        computePass.setBindGroup(0, gaussianBindGroup);
        computePass.setBindGroup(1, sortBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size), 1, 1);
        computePass.end();
      }

      // Preprocess assigns the sort.num_keys
      encoder.copyBufferToBuffer(
        sorter.sort_info_buffer,
        0,
        indirectDrawBuffer,
        // First u32 uses 4 bytes, need to offset to write to that stuff
        4,
        // Need to write 4 bytes, size of u32 atomic keys_size
        4
      );

      // Sort pass using preprocessed sort information
      sorter.sort(encoder);

      // TODO: Verify that this actually works
      render(encoder, texture_view);
    },
    camera_buffer,
  };
}

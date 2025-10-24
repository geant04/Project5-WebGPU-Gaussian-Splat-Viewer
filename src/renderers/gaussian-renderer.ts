import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import gaussianWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {

}

// Utility to create GPU buffers
// const createBuffer = (
//   device: GPUDevice,
//   label: string,
//   size: number,
//   usage: GPUBufferUsageFlags,
//   data?: ArrayBuffer | ArrayBufferView
// ) => {
//   const buffer = device.createBuffer({ label, size, usage });
//   if (data) device.queue.writeBuffer(buffer, 0, data);
//   return buffer;
// };

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

  // Somehow we need to know the number of splats there are, no?
  // This needs to be updated dynamically by the GPU culling stage
  const numSplats = pc.num_points;
  const floatsPerSplat = 4 + 4;
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

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const gaussian_shader = device.createShaderModule({code: gaussianWGSL });
  let preprocess_pipeline;
  let gaussian_bind_group;
  let splat_bind_group;
  let sort_bind_group;

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
        }
      ]
    });

    gaussian_bind_group = device.createBindGroup({
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
        }
      ]
    });

    
    const sortBindGroupLayout = device.createBindGroupLayout({
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

    sort_bind_group = device.createBindGroup({
      label: 'sort bind group',
      layout: sortBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
        { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
        { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
        { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
      ],
    });

    preprocess_pipeline = device.createComputePipeline({
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
  const splatPipeline = device.createRenderPipeline({
    label: 'render',
    layout: 'auto',
    vertex: {
      module: gaussian_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: gaussian_shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {

    // Need to dispatch the compute pass.
    // For now, this shouldn't do anything meaningful, and if it's wrong then we deal with that later
    if (testPreProcess)
    {
      const computePass = encoder.beginComputePass({
        label: 'pre-process compute pass'
      });

      computePass.setPipeline(preprocess_pipeline);
      computePass.setBindGroup(0, gaussian_bind_group);
      computePass.setBindGroup(1, sort_bind_group);
      computePass.dispatchWorkgroups(Math.ceil(pc.num_points / C.histogram_wg_size), 1, 1);
      computePass.end();
    }
    

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

    // Our only buffers to care about are:
    // - Camera buffer
    // - Gaussian 3D buffer

    // indirect draw thing mi bombaclat
    pass.drawIndirect(indirectDrawBuffer, 0);

    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // TODO: Figure out how this sorter thing works
      sorter.sort(encoder);

      // TODO: Verify that this actually works
      render(encoder, texture_view);
    },
    camera_buffer,
  };
}

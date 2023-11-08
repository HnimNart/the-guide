# 👨🏼‍💻 Exercises
Describe the base architecture of the egui-winit-wgpu template. Found in
```m2_concurrency::code::egui-winit-wgpu-template``` or
[online](https://github.com/absorensen/the-guide/tree/main/m2_concurrency/code/egui-winit-wgpu-template).

Which elements are in play?  
Who owns what data?  
Think back to what you have learned in this and the previous module.  
Use words and diagrams!  

# 🧬 Descriptions
Pick items worth a total of 3 points or more, write an interpretation of each
item of at least 10 times the number of points lines. So an item worth 2 points
requires a 20 line description.

Suggestions for things to talk about:

* A description of the proposed solution
* Which elements you have learned about in ```m1``` and ```m2``` are at play?
* What performance implications result from the item?
* What needs to be bottlenecked for this technique to be relevant (if it is an optimization technique)
* What will likely be the bottleneck after this technique has been implemented?
* What is the weakness of the method/design?
* In which cases would the proposed method/design be less useful?

You don't need to be correct, in many cases you can't be without profiling. The point is the process of verbalizing
analysis from a systems programming perspective.

## General

* 1 - Data-oriented design - Entity component systems
* 1 - Array of Structs, Structs of Arrays, Auto-Vectorization
* 1 - [Branch Prediction](https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array)
* 1 - [Eytzinger Binary Search](https://algorithmica.org/en/eytzinger)
* 2 - A [Mandelbrot](https://github.com/ProgrammingRust/mandelbrot/) program served 5 different ways
* 2 - Custom memory allocators
* 2 - [SIMD optimization](https://ipthomas.com/blog/2023/07/n-times-faster-than-c-where-n-128/)

## 🧬 Deep Learning

* 1 - PyTorch - Data-Distributed-Parallelism
* 1 - PyTorch - Model-Distributed-Parallelism
* 1 - PyTorch - [Optimizing inference](https://pytorch.org/blog/optimizing-libtorch/?hss_channel=lcp-78618366)
* 2 - Flash Attention
* 2 - Gyro Dropout - MLSys 2022
* 2 - [JAX](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
* 2 - [Fast as CHITA: Neural Network Pruning with Combinatorial Optimization](https://arxiv.org/abs/2302.14623)

## 🧬 Computer Graphics

* 1 - Linearized octrees
* 1 - Hierarchical Frustum Culling
* 2 - Sorting kernels in divergent workloads - Wavefront path tracing
* 2 - Shadertoy
* 2 - [Work Graphs in DX12](https://devblogs.microsoft.com/directx/d3d12-work-graphs-preview/)
* 4 - Nanite

## 🧬 Computer Vision

* 4 - ORB-SLAM - design and a warning about trying to code it

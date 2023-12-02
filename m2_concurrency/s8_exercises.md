# 👨🏼‍💻 Exercises

## m2::e0 - Archiectural Analysis
Describe the base architecture of the egui-winit-wgpu template. Found in
```m2_concurrency::code::egui-winit-wgpu-template``` or
[online][0].

Which elements are in play?  
Who owns what data?  
Think back to what you have learned in this and the previous module.  
Use words and diagrams!  

# 🧬 m2::e1 - Interpretation
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

You don't need to be correct, in many cases you can't be without profiling. The point is the process of putting
into words analysis from a systems programming perspective.

## General

* 1 - Entity component systems - [post 1][1], [post 2][2]
* 1 - Array-of-Structs, Structs-of-Arrays, Arrays-of-Structs-of-Arrays, Auto-Vectorization - [blog post][3]
* 1 - [Branch Prediction][4]
* 1 - [Eytzinger Binary Search][5]
* 2 - [Custom memory allocators][6]
* 2 - [SIMD optimization][7]

## 🧬 Deep Learning

* 1 - Data Distributed Parallelism - [Post 1][8], [Post 2][9]
* 1 - [Model Distributed Parallelism][10]
* 1 - [Optimizing Inference][11]
* 2 - [BAGUA: Scaling up Distributed Learning with System Relaxations][12]
* 2 - [Flash Attention][13]
* 2 - [Gyro Dropout][14], [Reference 2][15]
* 2 - [JAX][16]
* 2 - [Fast as CHITA: Neural Network Pruning with Combinatorial Optimization][17]
* 2 - [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models][22]

## 🧬 Computer Graphics

* 1 - [multiresolution ambient occlusion][24]
* 1 - [Fast Hierarchical Culling][18]
* 1 - [Octree Textures on the GPU][19]
* 2 - [On Ray Reordering Techniques for Faster GPU Ray Tracing][20]
* 2 - [Mesh Compression][21]
* 2 - [Work Graphs in DX12][23]
* 4 - Nanite - [Video 1][25], [Video 2][26], [Video 3][27]

## 🧬 Computer Vision

* 4 - ORB-SLAM - [Paper 1][28], [Paper 2][29], [Paper 3][30], [Simply Explained][31]

[0]: https://github.com/absorensen/the-guide/tree/main/m2_concurrency/code/egui-winit-wgpu-template
[1]: https://austinmorlan.com/posts/entity_component_system/
[2]: https://www.simplilearn.com/entity-component-system-introductory-guide-article
[3]: https://www.rustsim.org/blog/2020/03/23/simd-aosoa-in-nalgebra/
[4]: https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster-than-processing-an-unsorted-array
[5]: https://algorithmica.org/en/eytzinger
[6]: https://screwjankgames.github.io/engine%20programming/2020/09/24/writing-your-own-memory-allocators.html
[7]: https://ipthomas.com/blog/2023/07/n-times-faster-than-c-where-n-128/
[8]: https://pytorch.org/tutorials/beginner/ddp_series_theory.html
[9]: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
[10]: https://pytorch.org/docs/stable/pipeline.html
[11]: https://pytorch.org/blog/optimizing-libtorch/?hss_channel=lcp-78618366
[12]: https://arxiv.org/pdf/2107.01499.pdf
[13]: https://arxiv.org/pdf/2205.14135.pdf
[14]: https://proceedings.mlsys.org/paper_files/paper/2022/file/72aa1632b83c93a2f680dbb5235f1a83-Paper.pdf
[15]: https://docs.mosaicml.com/projects/composer/en/stable/method_cards/gyro_dropout.html
[16]: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
[17]: https://arxiv.org/abs/2302.14623
[18]: https://cesium.com/blog/2015/08/04/fast-hierarchical-culling/
[19]: https://developer.nvidia.com/gpugems/gpugems2/part-v-image-oriented-computing/chapter-37-octree-textures-gpu
[20]: https://meistdan.github.io/publications/raysorting/paper.pdf
[21]: https://iquilezles.org/articles/meshcompression/
[22]: https://arxiv.org/pdf/2310.16795.pdf
[23]: https://devblogs.microsoft.com/directx/d3d12-work-graphs-preview/
[24]: https://iquilezles.org/articles/multiresaocc/
[25]: https://www.youtube.com/watch?v=TMorJX3Nj6U
[26]: https://www.youtube.com/watch?v=NRnj_lnpORU
[27]: https://www.youtube.com/watch?v=eviSykqSUUw
[28]: https://arxiv.org/pdf/1502.00956.pdf
[29]: https://arxiv.org/pdf/1610.06475v1.pdf
[30]: https://arxiv.org/pdf/2007.11898.pdf
[31]: https://www.uio.no/studier/emner/matnat/its/TEK5030/v20/forelesninger/lecture_10_3_orb-slam.pdf

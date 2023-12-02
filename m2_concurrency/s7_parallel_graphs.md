# 3️⃣ Parallel Graphs
How to efficiently use graphs in parallel is more of an advanced topic and poses [a wide range of challenges][3].
If your graph is read-only and reasonably, sized you can optimize your performance by reducing the amount of
pointer jumps you do and increase your cache coherence. One possible avenue is to construct your graph
through indices.

However, not only is it hard to construct graphs in Rust, but it's also hard to construct dynamic graphs. And even
hard to construct massive graphs which can be dynamically manipulated in parallel. If you are working on a single
node at a time, things become quite simple in parallel. You can wrap each node in a lock, such as a mutex. But,
it is quite common to have to process bigger and bigger neighbourhoods around vertices. One way to do this
is to alternative between read and write stages. In one stage, each thread reads a relevant neighbourhood, does
some processing, and adds changes to a list. Once all changes have been proposed, one or more threads can execute
the changes, resulting in a new graph. One such solution is described [here][2].

Sampling and repacking extremely huge graphs for training graph neural networks on GPUs is also a
research topic [on its own][1]. Meshes, widely used in graphics, is another type of graph. In order
to maintain spatial coherency, and improve performance, another active field is how to [best sort meshes][5]
(graphs) and choosing the [right sizes][6] into cohesive meshlets (neighbourhoods), which can be
culled individually instead of the entire mesh being culled. Check out this general intro to
[parallel graph processing][4].  

[1]: https://proceedings.mlsys.org/paper_files/paper/2022/file/afacc5db3e0e85b446e6c7727cd7dca5-Paper.pdf
[2]: https://www.researchgate.net/publication/354065094_Practical_Spatial_Hash_Map_Updates
[3]: https://www.sandia.gov/app/uploads/sites/210/2022/05/graphs-and-machines.pdf
[4]: https://gfxcourses.stanford.edu/cs149/fall22/lecture/graphsdram/
[5]: https://github.com/zeux/meshoptimizer
[6]: https://zeux.io/2023/01/16/meshlet-size-tradeoffs/

# 👨🏼‍💻 Exercises

## m1::e0 - Vector Memory
Write out the stack and heap memory of THIS sequence of vector operations.
You can represent unitialized memory with a *.  
The vector must double its capacity and copy over each element once it gets a push operation while at capacity.
The vector must resize to half its size and copy over each active element once it gets a pop operation while at
quarter capacity.

* Create new vector of capacity 4
* .push(5)
* .push(2)
* .push(1)
* .pop()
* .push(42)
* .push(350)
* .push(1337)
* .push(1)
* .push(2)
* .push(5)
* .push(10)
* .pop()
* .pop()
* .pop()
* .pop()
* .pop()
* .pop()
* .pop()
* .pop()

Barring a few errors here and there this should be a simple exercise. Except.
Are you sure you got all of the undefined (*) values right?

Which common operation could have replaced the last sequence of .pop() operations
in the case where you wouldn't be using the popped values?

## m1::e1 - Linearized Indexing  
You have an array of dimensions ```MN``` in a linearized array ```data```, write pseudo code that iterates
through all data elements using two for-loops and doubles the value in place.

You have an array of dimensions ```MNK``` in a linearized array ```data```, write pseudo code that iterates
through all data elements using three for-loops and triples the value in place.

## m1::e2 - Killing the Garbage Collector
<figure markdown>
![Image](../figures/harass_the_garbage_collector.png){ width="500" }
<figcaption>
Where would you add a pointer to hurt garbage collection the most?
</figcaption>
</figure>

Adding which pointer would result in the most nodes not being properly garbage collected?  
If the garbage collector implements cycle detection to depth 2 adding which pointer would break it?
The nodes can't point to themselves.  

<figure markdown>
![Image](../figures/general_graph.png){ width="500" }
<figcaption>
How could you make this sort of general graph, with very few restrictions, safe for garbage collection?
</figcaption>
</figure>

In the case of the multiply connected nodes, can you come up with a structural solution which allows
us to make arbitrary graphs in a garbage collected setting or safe in a C++/Rust setting?

## GPU Programming
This serves as both one of the few instances where we can explicitly program parts of the memory hierarchy
(shared memory resides in the L1 cache), but also as an introduction to parallel thinking, which you will
need in the next module.

### m1::e3 - 1D Convolution
Go to ```m1_memory_hierarchies::code::gpu_hand_in``` and follow the instructions.
You need to do 3 different version of 1D convolution, a naive version, one with zero padding of the input
signal and one with optimal usage of shared memory.

### m1::e4 - Matrix Multiplication
For the second section you need to do a matrix multiplication and a tiled matrix multiplication
(using shared memory). You should also try out doing a padded version.

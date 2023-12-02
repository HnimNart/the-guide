# 👨🏼‍💻 Exercises

## m0::e0
In order to get a hang of the basic of Rust I recommend doing [Advent of Code][1]
problems until you feel comfortable moving forward.

## m0::e1
You won't be confronted much with the borrow checker in these small problems. To get better practice structure,
smart pointers, dynamic dispatch and traits, I recommend moving on to doing [Ray Tracing in One Weekend][2] in Rust.
There is a code snippet which shows your image on screen instead of writing it to file in
```m1_memory_hierarchies::code::image_on_screen```.

Once you have completed this part 1 out of 3, I suggest modifying the resulting code in the following ways -

* Remove the use of ```dyn```. (Hint: Use enums)
* Remove the use of smart pointers.
(Hint: My solution was to use indices and a geometry service provided by dependency injection)
* Parallelize computation of pixels through ```rayon```.

[1]: https://adventofcode.com/
[2]: https://raytracing.github.io/books/RayTracingInOneWeekend.html

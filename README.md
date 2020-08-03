# FastNMS
Fast and simple implementation of Non Maximum Suppression algorithm for bound boxes selection within object detection domain.

It's made to be fast and simple. For polygons, in this case 4-gons clipping, the Clipper library was used and the rest is just vanilla C++ with STL dependencies.

In my personal benchark there was a ~600 speed up in time converting the original version from Python to C++. The execution time was measured using regular chrono library for x64 Release build (with compiler options set for maximum optimization for speed).

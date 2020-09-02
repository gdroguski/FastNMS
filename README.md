# FastNMS
Fast and simple implementation of Non Maximum Suppression algorithm for bounding boxes selection within the object detection domain, after the NN predictions are made.

It was intented to be fast and simple. For clipping polygons, in this case 4-gons clipping, the Clipper library was used and the rest is just vanilla C++ with few STL dependencies and dynamically allocated arrays instead of list or vectors for speed.

In my tiny personal benmchark there was a ~600 speed up in time, converting the original version from Python to C++. The execution time was measured using regular chrono library for x64 Release build (with compiler options set for maximum optimization for speed).

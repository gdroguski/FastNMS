# FastNMS
Fast and simple implementation of Non-Maximum Suppression algorithm for bounding boxes selection within the object detection domain, after the NN predictions are made.

It was intented to be fast and simple. For clipping polygons, in this case 4-gons clipping, the Clipper library was used and the rest is just vanilla C++ with few STL dependencies and dynamically allocated arrays instead of list or vectors for speed.

In my tiny personal benchmark there was a ~600 speed up in time, converting the original version from Python to C++. The execution time was measured using regular chrono library for x64 Release build (with compiler options set for maximum optimization for speed).

I added also a `DLL` version of this mini project to be portable or 'runnable' from python - look below.

## How run it in C++:

Just run `main.cpp` with `nms_boxes.txt` input defined. Input with lines withe the style: `x1, y1, x2, y2, x3, y3, x4, y4, char_proba`

## How to use it with Python:

Header for this DLL:
```cpp
extern "C" NMSCLIPPER_API int nms(
	const float* boxes, const int& n, const int& m,
	float* new_boxes, int* pick,
	const double& overlapThresh, const double& neighbourThresh,
	const float& minScore, const int& num_neig
);

extern "C" NMSCLIPPER_API int nmsWithCharCls(
	const float* char_boxes, const int& n, const int& m,
	float* new_char_boxes, int* pick,
	const float* char_scores, const int& nc, const int& mc,
	float* new_char_scores,
	const double& overlapThresh, const double& neighbourThresh,
	const float& minScore, const int& num_neig
);
```

It's intended to be used in Python 3 with ctypes. Full Python example for standard NMS:

```python
# Imports
import numpy as np
from ctypes import c_float, c_int, c_double, POINTER, byref

# Loading DLL
cpp_dll = ctypes.cdll.LoadLibrary(path_to_lib)
cpp_nms = c_dll.nms

boxes = "some_numpy_array_with_bounding_boxes".astype(np.float32)
# 0 based shape is: n (instances) x 8 (x1, y1, x2, y2, x3, y3, x4, y4, nn_score)
n, m = boxes.shape
new_boxes = np.zeros_like(boxes).astype(np.float32)
pick = (np.zeros(n) - 1).astype(np.int)
        
c_float_p = POINTER(c_float)
c_int_p = POINTER(c_int)

# Conversion to C types
boxes_p = boxes.astype(np.float32).ctypes.data_as(c_float_p) # create dynamic array with floats from boxes
n_c = c_int(n)
m_c = c_int(m)
new_boxes_p = new_boxes.ctypes.data_as(c_float_p)
pick_p = pick.ctypes.data_as(c_int_p)
overlapThresh_c = c_double(0.15) # these are arbitraly
neighbourThresh_c = c_double(0.5)
minScore_c = c_float(0)
num_neig_c = c_int(1)

res = cpp_nms_ptr(
    boxes_p, byref(n_c), byref(m_c), 
    new_boxes_p, pick_p, 
    byref(overlapThresh_c), byref(neighbourThresh_c), byref(minScore_c), byref(num_neig_c)
)
    
if res != 0:
    raise RuntimeError('Unexpected error')
    
# Conversion from C types
new_boxes = np.array(new_boxes_p[:n*m]).reshape(n,m) # convert dynamic array to np.ndarray
pick = pick_p[:n] # convert dynamic array to list
pick = [p for p in pick if p != -1] # filter out unnecessary picks
```

---
layout: post
title: Applied Data Science with Python - Part 2 - Numpy
date: 2024-04-21 08:13 -0500
tags: [books, ds]
categories: [DS, Book-ADWP]
---
# Numpy

Numerical Python library

#### Creating Arrays

```python
import numpy as np

# Creating arrays
mylist = [1, 2, 3, 4, 5]
x = np.array(mylist)
print(x)

y = np.array([6, 7, 8, 9, 10])
print(y)
print(y.shape)

m = np.array([[1, 2, 3], [4, 5, 6]])
print(m)
print(m.shape)

# arange - produces an array of numbers from 0 to 30 with a step of 2
n = np.arange(0, 30, 2)
print(n)

# reshape - reshapes the array into a 3x5 matrix
n = n.reshape(3, 5)
print(n)

# linspace - produces an array of 9 numbers from 0 to 4
o = np.linspace(0, 4, 9)
print(o)

# resize - resizes the array to a 3x3 matrix
o.resize(3, 3)
print(o)

# ones - produces an array of 1s
print(np.ones((3, 2)))

# zeros - produces an array of 0s
print(np.zeros((2, 3)))

# eye - produces an identity matrix
print(np.eye(3))

# diag - produces a diagonal matrix
print(np.diag(y))

# repeat - repeats the array
print(np.array([1, 2, 3] * 3))
print(np.repeat([1, 2, 3], 3))

# Combining arrays
p = np.ones([2, 3], int)

# vstack - stacks arrays vertically
print(np.vstack([p, 2*p]))

# hstack - stacks arrays horizontally

print(np.hstack([p, 2*p]))
```

Output:

```
[1 2 3 4 5]
[ 6  7  8  9 10]
(5,)
[[1 2 3]
 [4 5 6]]
(2, 3)
[ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28]
[[ 0  2  4  6  8]
 [10 12 14 16 18]
 [20 22 24 26 28]]
[0.  0.5 1.  1.5 2.  2.5 3.  3.5 4. ]
[[0.  0.5 1. ]
 [1.5 2.  2.5]
 [3.  3.5 4. ]]
[[1. 1.]
 [1. 1.]
 [1. 1.]]
[[0. 0. 0.]
 [0. 0. 0.]]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[ 6  0  0  0  0]
 [ 0  7  0  0  0]
 [ 0  0  8  0  0]
...
 [2 2 2]
 [2 2 2]]
[[1 1 1 2 2 2]
 [1 1 1 2 2 2]]
```

#### Operations

```python
# Operations
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** 2)

# Dot product
print(x.dot(y))

# Transposing
z = np.array([y, y**2])
print(z)
print(z.shape)
print(z.T)
print(z.T.shape)
```

Output:

```
[ 7  9 11 13 15]
[-5 -5 -5 -5 -5]
[ 6 14 24 36 50]
[0.16666667 0.28571429 0.375      0.44444444 0.5       ]
[ 1  4  9 16 25]
130
[[  6   7   8   9  10]
 [ 36  49  64  81 100]]
(2, 5)
[[  6  36]
 [  7  49]
 [  8  64]
 [  9  81]
 [ 10 100]]
(5, 2)
```

#### Data types

```python
a = np.array([-4, -2, 1, 3, 5])
print(a.sum())
print(a.max())
print(a.min())
print(a.mean())
print(a.std())
print(a.argmax())
print(a.argmin())
```

Output:

```
3
5
-4
0.6
3.2619012860600183
4
0
```

#### Indexing and Slicing

```python
s = np.arange(13)**2
print(s)

print(s[0], s[4], s[0:3], s[1:5])

print(s[-4:])
print(s[-5::-2])

r = np.arange(36)
r.resize((6, 6))
print(r)

print(r[2, 2])
print(r[3, 3:6])

print(r[:2, :-1])
print(r[-1, ::2])

print(r[r > 30])
r[r > 30] = 30
print(r)

```

Output:

```
[  0   1   4   9  16  25  36  49  64  81 100 121 144]
0 16 [0 1 4] [ 1  4  9 16]
[ 81 100 121 144]
[64 36 16  4  0]
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]
 [30 31 32 33 34 35]]
14
[21 22 23]
[[ 0  1  2  3  4]
 [ 6  7  8  9 10]]
[30 32 34]
[31 32 33 34 35]
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]
 [30 30 30 30 30 30]]
```

#### Copying Data

```python
r2 = r[:3, :3]
print(r2)

r2[:] = 0
print(r2)

print(r)

r_copy = r.copy()
r_copy[:] = 10
print(r_copy, '\n')
print(r)
```

Output:

```
[[0 0 0]
 [0 0 0]
 [0 0 0]]
[[0 0 0]
 [0 0 0]
 [0 0 0]]
[[ 0  0  0  3  4  5]
 [ 0  0  0  9 10 11]
 [ 0  0  0 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]
 [30 30 30 30 30 30]]
[[10 10 10 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]
 [10 10 10 10 10 10]] 

[[ 0  0  0  3  4  5]
 [ 0  0  0  9 10 11]
 [ 0  0  0 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]
 [30 30 30 30 30 30]]
```

#### Iterating 

```python
test = np.random.randint(0, 10, (4, 3))
print(test)

for row in test:
    print(row)

for i in range(len(test)):
    print(test[i])

for i, row in enumerate(test):
    print('row', i, 'is', row)

test2 = test**2
print(test2)

for i, j in zip(test, test2):
    print(i, '+', j, '=', i+j)

```

Output:

```
[[7 5 0]
 [0 6 3]
 [9 1 3]
 [4 6 9]]
[7 5 0]
[0 6 3]
[9 1 3]
[4 6 9]
[7 5 0]
[0 6 3]
[9 1 3]
[4 6 9]
row 0 is [7 5 0]
row 1 is [0 6 3]
row 2 is [9 1 3]
row 3 is [4 6 9]
[[49 25  0]
 [ 0 36  9]
 [81  1  9]
 [16 36 81]]
[7 5 0] + [49 25  0] = [56 30  0]
[0 6 3] + [ 0 36  9] = [ 0 42 12]
[9 1 3] + [81  1  9] = [90  2 12]
[4 6 9] + [16 36 81] = [20 42 90]
```

```python
import numpy as np    # it is an unofficial standard to use np for numpy
import time

L = [1,2,3]
A = np.array([1,2,3])
for e in A:
    print(e)

A + np.array([4]) # broadcasting
A + np.array([4,5,6]) # adding vectors of same length
#A + np.array([4,5]) # error 

print(2*A)

data = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
print(data)
print(data*10)
print(data + data)
print(data.shape)
print(data.dtype)
m = np.array([[1, 2, 3], [4, 5, 6]])
print(m.dtype)

mylist = [1, 2, 3, 4, 5]
x = np.array(mylist)
print(x.shape)

np.empty((2, 3, 2))
np.ones((2, 3, 2))
print(np.eye(3))


arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)
arr = arr.astype(np.float32)
print(arr.dtype)

arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print(arr * arr)
print(arr - arr)
print(1 / arr)
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
print(arr2 > arr)

arr = np.arange(10)
print(arr)
print(arr[5:8])
arr[5:8] = 12
print(arr)
arr_slice = arr[5:8]
print(arr_slice)
arr_slice[1] = 12345
print(arr)
arr_slice[:] = 64
print(arr)

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[0, 2] # same as arr2d[0][2]
print(arr2d[:2, 1:]) # first two rows, last two columns
print(arr2d[:, :1]) # first column

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = [u{a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce an error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     
print("a[2:7:1] = ", c)

# access 3 elements separated by two 
c = a[2:7:2];     
print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        
print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        
print("a[:3]    = ", c)

# access all elements
c = a[:];         
print("a[:]     = ", c)

a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
b = -a 
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")

a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

#try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

a = np.array([1, 2, 3, 4])

# multiply a by a scalar
b = 5 * a 
print(f"b = 5 * a : {b}")
```
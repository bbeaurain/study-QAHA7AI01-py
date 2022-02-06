# Quick sheet

``` python
# Data types
np.array
np.zeros
np.ones
np.full
np.arange
np.linspace
np.random.random
np.random.normal
np.random.randint
np.eye
np.empty

#NumPy Arrays
.dim
.shape
.size
.dtype
.itemsize
.nbytes
x[n]
x[-n]
x[n,m]
x[start:stop:step]
.copy()
.reshape()
.newaxix
.concatenate()
.vstack()
.hstack()
.split()
.vsplit
.hsplit
.absolute()
.add.reduce()
.multiply.reduce()
.add.accumulate()
.multiply.accumulate()
.multiply.outer()
```

# Details
## Data Types
### A Python Integer is more than just an integer

### A Python List is more than just a list

### Fixed type arrays in Python

### Creating Arrays from Python Lists

Unlike Python lists, NumPy is constrained to arrays that all contain
the same type.

If we want to explicitly set the data type of the resulting array, we can use the dtype keyword:

``` python
np.array([1, 2, 3, 4], dtype='float32')
```
Unlike Python lists, NumPy arrays can explicitly be multidimensional

### Creating Arrays from Scratch

``` python
## Create a length-10 integer array filled with zeros
np.zeros(10, dtype=int)

## Create a 3x5 floating-point array filled with 1s
np.ones((3, 5), dtype=float)

## Create a 3x5 array filled with 3.14
np.full((3, 5), 3.14)

## Create an array filled with a linear sequence Starting at 0, ending at 20, stepping by 2 (this is similar to the built-in range() function)
np.arange(0, 20, 2)

## Create an array of five values evenly spaced between 0 and 1
np.linspace(0, 1, 5)

## Create a 3x3 array of uniformly istributed random values between 0 and 1
np.random.random((3, 3))

# Create a 3x3 array of normally distributed random values with mean 0 and standard deviation 1
np.random.normal(0, 1, (3, 3))

# Create a 3x3 array of random integers in the interval [0, 10)
np.random.randint(0, 10, (3, 3))

# Create a 3x3 identity matrix
np.eye(3)

# Create an uninitialized array of three integers
# The values will be whatever happens to already exist at that memory location
np.empty(3)
```
### NumPy Standard Data Types

https://numpy.org/doc/stable/user/basics.types.html

## Basics of Numpy Arrays

### Numpy Array Attributes

``` python
# One-dimensional array
x1 = np.random.randint(10, size=6) 
# Two-dimensional array
x2 = np.random.randint(10, size=(3, 4)) 
# Three-dimensional array
x3 = np.random.randint(10, size=(3, 4, 5)) 

# Number of dimension
print("x3 ndim: ", x3.ndim)
# Shape
print("x3 shape:", x3.shape)
# Size
print("x3 size: ", x3.size)
#Type
print("dtype:", x3.dtype)
# size of each array elements
print("itemsize:", x3.itemsize, "bytes")
# size of total array
print("nbytes:", x3.nbytes, "bytes")
```

### Array Indexing: Accessing Single Elements

``` python
x1[0]
x1[4]
```

### Array Indexing: Accessing Subarrays
``` python
# Single dimension
x[:5] # first five elements
x[5:] # elements after index 5
x[4:7] # middle subarray
x[::2] # every other element
x[1::2] # every other element, starting at index 1
x[::-1] # all elements, reversed
x[5::-2] # reversed every other from index 5

# Multi-dimensions
x2[:2, :3] # two rows, three columns
x2[:3, ::2] # 3 rows, every other column

# Accessing rows & columns
print(x2[:, 0]) # first column of x2
print(x2[0, :]) # first row of x2
print(x2[0]) # equivalent to x2[0, :]
```

Sub arrays are not copies from original arrays. If need to copy, use .copy()

``` python
x2_sub_copy = x2[:2, :2].copy()
```
### Reshaping of Array

``` python
grid = np.arange(1, 10).reshape((3, 3)) #put the numbers 1 through 9 in a 3×3 grid

########
x = np.array([1, 2, 3])
x.reshape((1, 3)) # row vector via reshape
x[np.newaxis, :] # row vector via newaxis
```
### Concatenation of Arrays

``` python
np.concatenate([x, y]) #concatenates x and y (can be multidimensional arrays)

concatenate along the first axis # np.concatenate([grid, grid])
np.concatenate([grid, grid], axis=1) # concatenate along the second axis (zero-indexed)

np.vstack([x, grid]) # vertically stack the arrays
np.hstack([grid, y]) # horizontally stack the arrays

x1, x2, x3 = np.split(x, [3, 5]) #split x into x1, x2, x3

upper, lower = np.vsplit(grid, [2]) #split vertically at the 2nd row
left, right = np.hsplit(grid, [2]) #split horizontally at 2nd column
```
## Computation on NumPy Arrays: Universal Functions
### The slowness of loops
### Introducing UFuncs
### Exploring UFuncs

```python
## Arithmetic

x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2) # floor division
print("-x = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2 = ", x % 2)
-(0.5*x + 1) ** 2

np.absolute(x) #absolute value

print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))

print("e^x =", np.exp(x))
print("2^x =", np.exp2(x))
print("3^x =", np.power(3, x))

print("ln(x) =", np.log(x))
print("log2(x) =", np.log2(x))
print("log10(x) =", np.log10(x))

print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))

print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2) =", special.beta(x, 2))

print("erf(x) =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))

## Advanced

np.multiply(x, 10, out=y) #multiplies x's values by 10 and stores result in y
np.power(2, x, out=y[::2]) 

## Aggregate

np.add.reduce(x) #results of addition of all elements
np.multiply.reduce(x) #multiplication of all elements
np.add.accumulate(x) # array of addition accumulation
np.multiply.accumulate(x) #array of multiplication accumulation
```
Operator|Equivalent ufunc|Description
--|--|--
+|np.add|Addition (e.g., 1 + 1 = 2)
-|np.subtract|Subtraction (e.g., 3 - 2 = 1)
-|np.negative|Unary negation (e.g., -2)
*|np.multiply|Multiplication (e.g., 2 * 3 = 6)
/|np.divide|Division (e.g., 3 / 2 = 1.5)
//|np.floor_divide|Floor division (e.g., 3 // 2 = 1)
**|np.power|Exponentiation (e.g., 2 ** 3 = 8)
%|np.mod|Modulus/remainder (e.g., 9 % 4 = 1)



## Aggregations: Min, Max, and everything in between

## Computation on Arrays: Broadcasting

## Comparisons, Masks and Boolean Logic

``` python
## how many values less than 6?
np.count_nonzero(x < 6)
np.sum(x < 6)

## how many values less than 6 in each row?
np.sum(x < 6, axis=1)

## are there any values greater than 8?
np.any(x > 8)

## are all values less than 10?
np.all(x < 10)

## are all values equal to 6?
np.all(x == 6)

## are all values in each row less than 8?
np.all(x < 8, axis=1)

## Boolean operators
np.sum((inches > 0.5) & (inches < 1))
np.sum(~( (inches <= 0.5) | (inches >= 1) ))
```
### Boolean operators and their equivalent
ufuncs:
Operator|Equivalent ufunc
---|---
&|np.bitwise_and
/(vertical)|np.bitwise_or
^|np.bitwise_xor
~|np.bitwise_not

## Fancy Indexing

``` python
x = rand.randint(100, size=10) ## Returns [51 92 14 71 60 20 82 86 74 74]

[x[3], x[7], x[2]] ## Returns [71, 86, 14]

## below returns array([71, 86, 60])
ind = [3, 7, 4]
x[ind]
```

## Sorting Arrays

### General
``` python
x = np.array([2, 1, 4, 3, 5])

## returns sorted list
np.sort(x) 
## returns indices of the sorted elements
i = np.argsort(x)

## sort each column of X
np.sort(X, axis=0)
## sort each row of X
np.sort(X, axis=1)
```

### Partial Sorts: Partitioning

``` python
## Returns the 3 smalest values of X to the left, and the remaining to the right
np.partition(x, 3)

## returns the indices instead of the resulting array
np.argpartition
```

## Structured Data: Numpy's structured arrays

``` python
## Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight')
```
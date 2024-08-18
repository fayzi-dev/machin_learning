# Example Numpy Code by Mohammad fayzi

# Install Numpy:Install it using this command:
# pip
# pip3
    pip install numpy
    pip3 install numpy
        
# Import Numpy : show version numpy
# NumPy is usually imported under the np alias.

    import numpy as np
    np.__version__

# Create a new arrray by Numpy
    arr = np.array([1, 2, 3, 4])
    # print(arr)
    print(arr.shape)


# Use a tuple to create a NumPy array:

    arr_tuple = np.array((1 ,2 ,3 , 4, 5, 6))
    print(arr_tuple)
 
# Create a NumPy ndarray Object

    arr_2_D = np.array([[10,  20, 30],[200, 300, 400]])
    print(arr_2_D)
        
# Get the Shape of an Array

    print(arr_2_D.shape)
 

#Access Array Elements 
# Array indexing is the same as accessing an array element
   arr = np.array([1, 2, 3, 4])

   print(arr[0])

# Slicing arrays برش آرایه ها
# Slicing in python means taking elements from one given index to another given index.

# We pass slice instead of index like this: [start:end].

# We can also define the step, like this: [start:end:step].
    arr_b = np.array([[1, 2, 3],[6, 7, 8],[9, 10, 11]])
    print(arr_b[1:2])

#Checking the Data Type of an Array

    arr_b = np.array([[1, 2, 3],[6, 7, 8],[9, 10, 11]])
    print(arr_b.dtype)
# Data Types in NumPy
# NumPy has some extra data types, and refer to data types with one character, like i for integers, u for unsigned integers etc.

# Below is a list of all data types in NumPy and the characters used to represent them.

# i - integer
# b - boolean
# u - unsigned integer
# f - float
# c - complex float
# m - timedelta
# M - datetime
# O - object
# S - string
# U - unicode string
# V - fixed chunk of memory for other type ( void )
# Create an array with data type String:
    arr_c = np.array(['ali', 'mohammad','hasan'],dtype='S')
    print(arr_c.dtype)

# Reshape Numpy
# Convert the following 1-D array with 12 elements into a 2-D array.

# The outermost dimension will have 4 arrays, each with 3 elements:
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    newarr = arr.reshape(4, 3)

    print(newarr)

# Convert the following 1-D array with 12 elements into a 3-D array.

# The outermost dimension will have 2 arrays that contains 3 arrays, each with 2 elements:
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    newarr = arr.reshape(2, 3, 2)
   
    print(newarr)


# Check if the returned array is a copy or a view:

    import numpy as np

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    arr.reshape(2, 4).base

# Iterate on the elements of the following 2-D array:

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    for x in arr:
    print(x)

# Joining NumPy Arrays

    arr1 = np.array([1, 2, 3])

    arr2 = np.array([4, 5, 6])

    arr = np.concatenate((arr1, arr2))

    print(arr)

# Iterating Arrays
# Iterating means going through elements one by one.

# As we deal with multi-dimensional arrays in numpy, we can do this using basic for loop of python.

# If we iterate on a 1-D array it will go through each element one by one.
    arr = np.array([1, 2, 3, 4, 5])
    
    for x in arr:
        print(x)

# 2-D Array
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    for x in arr:
    for y in x:
        print(y)

# Split the 2-D array into three 2-D arrays.
    arr = np.array([1, 2, 3, 4, 5, 6])

    newarr = np.array_split(arr, 3)

    print(newarr[0])
    print(newarr[1])
    print(newarr[2])


# Searching Arrays
# You can search an array for a certain value, and return the indexes that get a match.

# To search an array, use the where() method.
    arr = np.array([1, 2, 3, 4, 5, 4, 4])

    x = np.where(arr == 4)

    print(x)

# Sorting Arrays
    arr = np.array([[3, 2, 4], [5, 0, 1]])

    print(np.sort(arr))

# Filter Array


    arr = np.array([41, 42, 43, 44])

    filter_arr = arr > 42

    newarr = arr[filter_arr]

    print(filter_arr)
    print(newarr)

#Generate Random Array
#Arange
    rand_3 = np.arange(1, 20, step=2)
    print(rand_3)
#Linspace
    rand_4 = np.linspace(1, 2 ,num=10)
    print(rand_4)
#import random 
    from numpy import random
#rand
    rand_arr = random.rand(3, 5)
    print(rand_arr)
#Randint
    rand_1 = random.randint(100, size=4)
    print(rand_1)
# Choice
    x = random.choice([3, 5, 7, 9], size=(3, 5))

    print(x)

#Operation Numpy Array

    arr_1 = random.rand(2, 2)
    arr_2 = random.rand(2,2)
    print(np.add(arr_1, arr_2))
    print(np.subtract(arr_1, arr_2))
    print(np.sqrt(arr_1, arr_2))
    print(np.divide(arr_1, arr_2))

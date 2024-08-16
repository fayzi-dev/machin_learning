# Machin learning

<!--  Install Numpy:Install it using this command:
 pip
 pip3 -->
    pip install numpy
    pip3 install numpy
        
<!--  Import Numpy : show version numpy
 NumPy is usually imported under the np alias. -->

    import numpy as np
    np.__version__

<!--  Create a new arrray by Numpy -->
    arr = np.array([1, 2, 3, 4])
    # print(arr)
    print(arr.shape)


<!-- # Use a tuple to create a NumPy array: -->

    arr_tuple = np.array((1 ,2 ,3 , 4, 5, 6))
    print(arr_tuple)
 
<!-- # ایجاد آرایه دو بعدی -->

    arr_2_D = np.array([[10,  20, 30],[200, 300, 400]])
    print(arr_2_D)
        
<!-- # (shape) برای نمایش تعداد بعد آرایه -->

    print(arr_2_D.shape)
 

<!-- # دسترسی به عناصر آرایه ها  -->
   arr = np.array([1, 2, 3, 4])

   print(arr[0])

<!-- # Slicing arrays برش آرایه ها
# Slicing in python means taking elements from one given index to another given index.

# We pass slice instead of index like this: [start:end].

# We can also define the step, like this: [start:end:step]. -->
    arr_b = np.array([[1, 2, 3],[6, 7, 8],[9, 10, 11]])
    print(arr_b[1:2])

#Checking the Data Type of an Array

    arr_b = np.array([[1, 2, 3],[6, 7, 8],[9, 10, 11]])
    print(arr_b.dtype)
<!-- # Data Types in NumPy
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
# Create an array with data type String: -->
    arr_c = np.array(['ali', 'mohammad','hasan'],dtype='S')
    print(arr_c.dtype)

<!-- # Reshape Numpy -->
<!-- # Convert the following 1-D array with 12 elements into a 2-D array. -->

<!-- # The outermost dimension will have 4 arrays, each with 3 elements: -->
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    newarr = arr.reshape(4, 3)

    print(newarr)

<!-- # Convert the following 1-D array with 12 elements into a 3-D array. -->

<!-- # The outermost dimension will have 2 arrays that contains 3 arrays, each with 2 elements: -->
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    newarr = arr.reshape(2, 3, 2)
   
    print(newarr)


<!-- # Check if the returned array is a copy or a view: -->

    import numpy as np

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    arr.reshape(2, 4).base

<!-- # Iterate on the elements of the following 2-D array: -->

    arr = np.array([[1, 2, 3], [4, 5, 6]])

    for x in arr:
    print(x)
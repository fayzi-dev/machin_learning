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
 
# ایجاد آرایه دو بعدی

    arr_2_D = np.array([[10,  20, 30],[200, 300, 400]])
    print(arr_2_D)
        
# (shape) برای نمایش تعداد بعد آرایه

    print(arr_2_D.shape)
 

# دسترسی به عناصر آرایه ها 
   arr = np.array([1, 2, 3, 4])

   print(arr[0])

# Slicing arrays برش آرایه ها
# Slicing in python means taking elements from one given index to another given index.

# We pass slice instead of index like this: [start:end].

# We can also define the step, like this: [start:end:step].
    arr_b = np.array([[1, 2, 3],[6, 7, 8],[9, 10, 11]])
    print(arr_b[1:2])
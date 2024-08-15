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
     
'1.26.4'


# Create a new arrray by Numpy


    arr = np.array([1, 2, 3, 4])
    # print(arr)
    print(arr.shape)
        
    (4,)


# Use a tuple to create a NumPy array:

    arr_tuple = np.array((1 ,2 ,3 , 4, 5, 6))
    print(arr_tuple)
        
    array([1, 2, 3, 4, 5, 6])
# ایجاد آرایه دو بعدی

    arr_2_D = np.array([[10,  20, 30],[200, 300, 400]])
    print(arr_2_D)
        
    [[ 10  20  30]
    [200 300 400]]
# (shape) برای نمایش تعداد بعد آرایه

    print(arr_2_D.shape)
        
    (2, 3)


# دسترسی به عناصر آرایه ها 
    arr = np.array([1, 2, 3, 4])

    print(arr[0])

    1
# Pandas is a powerful Python library for data manipulation and analysis.
#  It provides high-performance, easy-to-use data structures and 1  data analysis tools.
#  Let's dive into the basics:


# Import Pandas
    import pandas as pd



#Series 

    # Create a Series
    data = [1, 2, 3, 4, 5]
    s = pd.Series(data)
    print(s)

# DataFrame

    # Create a DataFrame
    data = {'Name': ['Ali', 'Mohammad', 'Zahra'],
            'Age': [25, 31, 19]}
    df = pd.DataFrame(data)
    print(df)

# Reading and Writing Data
# Pandas can read data from various file formats like CSV, Excel, JSON, etc.,
# and write data to these formats as well.
# 1.
    # Read a CSV file
    df = pd.read_csv('data.csv')

    # Write a DataFrame to a CSV file
    df.to_csv('output.csv', index=False)
# 2.
    df = pd.read_csv('titanic.csv')
    df.iloc[:,[1,4,8]]
    df.to_csv('output.csv', index=False)


# Selecting data:

    # Select a column
    print(df['Name'])

    # Select rows by index
    print(df.iloc[0])

    # Select rows by condition
    print(df[df['Age'] > 25])

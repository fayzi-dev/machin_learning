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


# Pandas - Analyzing DataFrames 

    print(df.head(10)) #Get a quick overview by printing the first 10 rows of the DataFrame
    print(df.tail()) #Print the last 5 rows of the DataFrame
    print(df.info()) #Print information about the data

# Pandas - Cleaning Data
    # Pandas - Cleaning Empty Cells
    df = pd.read_csv('data.csv')

    df.dropna(inplace = True)

    print(df) 

    # Replace NULL values in the "Age" columns with the number 50:
    df["Age"].fillna(50,inplace=True)
    print(df.info())


    # Pandas - Fixing Wrong Data

        #  Replace value
        for x in df.index:
          if df.loc[x, "Duration"] > 12:
           df.loc[x, "Duration"] = 12

        # Remove Rows
        for x in df.index:
          if df.loc[x, "Duration"] > 12:
            df.drop(x, inplace = True)

        # Removing Duplicates
        dup_df = df.loc[1:20,["Age"]]
        # dup_df.duplicated()
        dup_df.drop_duplicates(inplace=True)
        print(dup_df)



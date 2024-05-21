import pandas as pd
import os
from tkinter import filedialog
from tkinter import Tk
import gzip


def load_glider_nav():
    ''' Load the navigation data from glider output files
        Parameter :
        Returns :
            df : dataframe with glider navigation data
    '''
    root = Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title='Select master folder')

    all_rows = []  # Initialize an empty list to store the contents of all CSV files
    yo = []  # List to store the file numbers
    file = []

    first_file = True
    file_number = 1  # Initialize the file number

    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            file_path = os.path.join(directory, filename)

            with gzip.open(file_path, 'rt') as gz_file:
                delimiter = ';'  # Specify the desired delimiter
                gz_reader = pd.read_csv(gz_file, delimiter=delimiter)
                # If it's the first file, append the header row
                if first_file:
                    all_rows.append(gz_reader.columns.tolist())
                    first_file = False
                # Add the rows from the current CSV file to the all_rows list
                all_rows.extend(gz_reader.values.tolist())
                # Add yo number to the yo list
                yo.extend([filename.split('.')[-2]] * len(gz_reader))
                file.extend([filename] * len(gz_reader))
                file_number += 1  # Increment the file number for the next file

    # Create a DataFrame from the combined data
    df = pd.DataFrame(all_rows, parse_dates=['Timestamp'])
    df.columns = df.iloc[0]  # set 1st row as headers
    df = df.iloc[1:, 0:-1]  # delete last column and 1st row

    # Add the yo number to the DataFrame
    df['yo'] = [int(x) for x in yo]

    df['file'] = file
    df = df.sort_values(by=['Timestamp'])
    df = df.drop(df[(df['Lat'] == 0) & (df['Lon'] == 0)].index).reset_index(drop=True)
    df['Lat DD'] = [int(x) + (((x - int(x)) / 60) * 100) for x in df['Lat'] / 100]
    df['Lon DD'] = [int(x) + (((x - int(x)) / 60) * 100) for x in df['Lon'] / 100]
    # df['Datetime'] = [pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S') for x in df['Timestamp']]
    df['Depth'] = -df['Depth']

    return df

# main.py
from file_io import read_csv, read_category_csv
from data_processing import categorize_table, merge_tables
from merge_tables import merge_tables
import csv

doLoop = True
while(doLoop)
    # Load CSV File
    FILE_PATH_MASTER = input("Filename for table 1: ") #'data.csv'
    FILE_PATH_DEPENDENT = input("Filename for table 2: ") #'data2.csv'
    FILE_PATH_CATEGORIES = 'categories_data.csv'
    
    master_table = read_csv(FILE_PATH_MASTER)
    dependent_table = read_csv(FILE_PATH_DEPENDENT)
    
    # Map where key is the category and value are the aliases used in tables
    # i.e temperature: {temperature,temp,degrees,celsius,fahrenheit,thermometer}
    category_aliases = read_category_csv(FILE_PATH_CATEGORIES)
    
    # Categorize the Columns
    master_categories = categorize_table(master_table, category_aliases)
    dependent_categories = categorize_table(dependent_table, category_aliases)
    
    # Merge the tables
    #merged_data = merge_tables(master_categories, dependent_categories, len(master_table) - 1, len(dependent_table) - 1)
    merged_table = merge_tables(master_table, dependent_table)
    
    # Write to CSV
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
    
        for row in merged_table:
            writer.writerow(row)
    
        print("Successfully printed to csv file")

    while(True)
        doAgain = input("Would you like to merge more tables? (Y/N)")
        if doAgain == "N":
            doLoop = False
            break
        if doAgain == "Y":
            doLoop = True
            break

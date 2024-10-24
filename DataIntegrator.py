#Libraries
import csv
import os


# In[4]:


# File paths for master and dependent data
FILE_PATH_MASTER = 'data.csv'
FILE_PATH_DEPENDENT = 'data2.csv'
NULL_REP = 'NULL'

def open_csv_file(file_path):
    master_data = []
    
    if os.path.exists(file_path):  # Check if the file exists
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            k = next(reader)
            length = len(k)
            file.seek(0)
            for row in reader:
                # Ensure the row has the correct length
                if len(row) < length:
                    row.append(NULL_REP)
                
                # Replace all empty strings '' with NULL_REP
                row = [NULL_REP if item == '' else item for item in row]
                
                master_data.append(row)

    else:
        # Create a new file if it does not exist
        print(f"The file '{file_path}' does not exist. Creating a new file.")
        with open(file_path, mode='w', newline=''):
            pass  # Just create an empty file

    return master_data

# Load master and dependent data
master_data = open_csv_file(FILE_PATH_MASTER)
dependent_data = open_csv_file(FILE_PATH_DEPENDENT)

# Optionally print out the loaded data for verification
print("Master Data:", master_data)
print("Dependent Data:", dependent_data)

master_row_count = len(master_data) - 1
dependent_row_count = len(dependent_data) - 1


# In[5]:


# Base class for all data fields
class DataField:
    def __init__(self, value):
        self.value = value
        self.class_name = self.__class__.__name__.lower() # Store class name automatically

    def display(self):
        raise NotImplementedError("Subclasses must implement this method.")

# Subclass for each specific category
class Name(DataField):
    def convert(self):
        self.value = [name.split()[0].capitalize() for name in self.value]

    def display(self):
        return f"{self.class_name}: {self.value}"

class Date(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class Time(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class Temperature(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}°C"

class Status(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class Address(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class ID(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class PhoneNumber(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class Email(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class Price(DataField):
    def display(self):
        return f"{self.class_name}: ${self.value}"

class Quantity(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class Age(DataField):
    def display(self):
        return f"{self.class_name}: {self.value} years"

class Gender(DataField):
    def display(self):
        return f"{self.class_name}: {self.value}"

class Weight(DataField):
    def display(self):
        return f"{self.class_name}: {self.value} kg"

class Height(DataField):
    def display(self):
        return f"{self.class_name}: {self.value} m"


# Factory for creating DataField instances
class DataFieldFactory:
    @staticmethod
    def create_data_field(field_type, value):
        field_classes = {
            "name": Name,
            "date": Date,
            "time": Time,
            "temperature": Temperature,
            "status": Status,
            "address": Address,
            "id": ID,
            "phone_number": PhoneNumber,
            "email": Email,
            "price": Price,
            "quantity": Quantity,
            "age": Age,
            "gender": Gender,
            "weight": Weight,
            "height": Height,
        }
        
        field_class = field_classes.get(field_type.lower())
        if not field_class:
            raise ValueError(f"Unknown data field type: {field_type}")
        
        return field_class(value)


# Example usage
def display_data(fields):
    for field in fields:
        print(field.display())


# In[6]:


# Threshold for categorizing column names
CATEGORY_THRESHOLD = 10

def get_column_values(file_data, index):
    """
    Retrieves all values from a specified column index in the given data.
    
    Args:
        file_data (list): 2D list containing data.
        index (int): The index of the column to retrieve.

    Returns:
        list: A list of values from the specified column.
    """
    column_values = []
    for row in file_data[1:]:  # Skip the header row
        column_values.append(row[index])
    return column_values

def levenshtein_distance(str1, str2):
    """
    Calculates the Levenshtein distance between two strings.
    
    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        int: The Levenshtein distance.
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i 
    for j in range(n + 1):
        dp[0][j] = j 

    # Compute the distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1 
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    
    return dp[m][n]  # Return the distance

def categorize_column(column_name, categories):
    """
    Categorizes a column based on the minimum Levenshtein distance 
    compared to known categories.
    
    Args:
        column_name (str): The name of the column to categorize.
        categories (list): A list of category names and variations.

    Returns:
        str: The category assigned to the column.
    """
    current_category = "etc"
    min_distance = CATEGORY_THRESHOLD

    for category_row in categories[1:]:  # Skip the header
        category_name = category_row[0]
        for other_name in category_row[1:]:
            distance = levenshtein_distance(other_name.lower(), column_name.lower())
            if distance < min_distance:
                min_distance = distance
                current_category = category_name
                
    return current_category

def categorize_all_columns(data_table, categories):
    """
    Categorizes all columns in the data table based on the defined categories.
    
    Args:
        data_table (list): The 2D list representing the data table.
        categories (list): The categories to compare against.

    Returns:
        dict: A dictionary where the key is the column name, and the value is the DataField instance.
    """
    table_categories = {}
    
    # Iterate over column names
    for j, column_name in enumerate(data_table[0]):
        current_category = categorize_column(column_name, categories)
        column_values = get_column_values(data_table, j)
        
        # Create a DataField instance for the categorized column and store it in the dictionary
        table_categories[current_category] = DataFieldFactory.create_data_field(current_category, column_values)

    print("Column Names:", data_table[0])
    print("Assigned Categories:", table_categories)
    return table_categories

# Example usage
categories = open_csv_file('categories_data.csv')
master_categories = categorize_all_columns(master_data, categories)
dependent_categories = categorize_all_columns(dependent_data, categories)

# Convert the first master category for demonstration
master_categories['name'].convert()
print('Processing completed.')


# In[7]:


# Takes the primary field and appends the secondary field, fills missing data with NULL_REP
def mergeTables(master_data, dependent_data):
    merged_data = master_data.copy()
    all_keys = list(master_data.keys())
    
    # Append keys from secondary_data that aren't in master_data
    all_keys.extend([column_key for column_key in dependent_data.keys() if column_key not in master_data])

    for column_key in all_keys:

        # if the master_data does not have a key, will fill in NULL
        if column_key not in master_data:
            null_filled_values = [NULL_REP for _ in range(master_row_count)]
            merged_data[column_key] = DataFieldFactory.create_data_field(column_key, null_filled_values)
        
        # Append depentent data if it exists for a key
        if column_key in dependent_data and dependent_data[column_key].value:
            merged_data[column_key].value.extend(dependent_data[column_key].value)
        # Append NULL_REP if no value exists in depentent_data
        else:

            merged_data[column_key].value.extend([NULL_REP for _ in range(dependent_row_count)])

    # Print all values for each key in the merged_data
    for column_key in merged_data:
        print(f"Key: {column_key}, Values: {merged_data[column_key].value}")
    
    return merged_data

merged_data = mergeTables(master_categories, dependent_categories)

# Write to csv file in column fashion
with open('databaseOutput.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write headers (keys)
    column_keys = list(merged_data.keys())
    writer.writerow(column_keys)

    # Determine the maximum number of rows (longest list in merged_data)
    max_length = max(len(merged_data[column_key].value) for column_key in column_keys)

    # Write rows by transposing the lists in merged_data
    for i in range(max_length):
        row = [merged_data[column_key].value[i] if i < len(merged_data[column_key].value) else NULL_REP for column_key in column_keys]
        writer.writerow(row)

print("Data successfully written to 'databaseOutput.csv' in column format.")


# In[ ]:





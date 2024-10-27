# read_table.py


# This function reads in the contents of a table formatted as a csv file
def readTable(filename):

    # Get file contents
    tableFile = open(filename)
    fileContents = []
    for row in tableFile:
        fileContents.append(row)
    tableFile.close()

    # Arrange file contents into a 2-D array
    tableArray = []	
    for row in fileContents:
        rowValues = row.split(',')
        newRow = []
        for value in rowValues:
            newRow.append(value.rstrip())
        tableArray.append(newRow)

    return tableArray
def remove_string_from_file(input_file, output_file, string_to_remove):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            modified_line = line.replace(string_to_remove, '')
            fout.write(modified_line)

# File paths
input_file = 'data_xfoil.csv'
output_file = 'data_xfoil_nonan.csv'

# String to remove
string_to_remove = 'nan'

# Remove the string from the file
remove_string_from_file(input_file, output_file, string_to_remove)

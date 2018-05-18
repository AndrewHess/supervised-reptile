'''
This program combines several classes into one class with a specified number
of data points. The new class is created by repeatedly randomly selecting one
of the input classes and adding the next data point in that class to combined
class.
'''

import random

path = 'data/kdd_diff_attacks.nosync/test/'
folders = ['back_400', 'neptune_400', 'portsweep_400', 'teardrop_400', 'ipsweep_400', 'satan_400', 'warezclient_400']
in_size = 400
out_size = 400
out_folder = 'data/kdd_binary.nosync/test/attack_400'
included = [] # A list of files included in the new class; used to prevent duplicate files

i = 0
while i < out_size:
    print('i:', i)

    # Select a random input folder and file.
    folder_ind = random.randint(0, len(folders) - 1)
    file_ind = random.randint(0, in_size - 1)

    file_ref = folders[folder_ind] + '/' + str(file_ind)

    # Check if this file is already included.
    if file_ref in included:
        i -= 1
        continue

    # Add this file to the new class
    included.append(file_ref)
    in_file  = open(path + folders[folder_ind] + '/data' + str(file_ind) + '.csv', 'r')
    out_file = open(out_folder + '/data' + str(i) + '.csv', 'w+')

    line = in_file.readline()
    out_file.write(line)

    in_file.close()
    out_file.close()

    i += 1

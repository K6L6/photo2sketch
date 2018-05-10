import ipdb
import pandas as pd
import os
import os.path

# filename = raw_input()
# ipdb.set_trace()

inv_svg = []
folder_p = 'Sketchy_data_valid/duck/'
filename = 'invalid.txt'
file_p = folder_p + filename
data = pd.read_csv(file_p)#, skip_blank_lines = False)
for i in range(0, len(data)):
    if 'n0' in data[data.columns[0]][i]:
        svg = data[data.columns[0]][i]+'.svg'
        inv_svg.append(svg)
    else:
        pass

# print(len(inv_svg))
# ipdb.set_trace()

for x in inv_svg:
    if os.path.isfile(folder_p + x):
        os.remove(folder_p + x)
    else:
        pass
    #print(x)

# print(inv_svg)
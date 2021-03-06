These scripts will require the magenta environment because they use `utils` which is used in sketch-rnn.

# svg_parser

This contains several functions which were used mostly for reading svg files. 

The two variables on the top of the script:
`folder_p` --> is the path of the folder which contains .svg files.
`dest_p` --> is the path of the folder where the .npz file will be created.

The first two fucntions `rsvg_in_folderxy` and `rsvg_in_folders3` are used to read a specified number of files within a specified path. These functions return a list which is a list of x-y coordinates for `rsvg_in_folderxy`, and a list in stroke-3 format for `rsvg_in_folders3`.
input arguments:
f_path --> path of the folder which contains .svg files.
no_of_files --> number of .svg files that you want to read from the specified folder.

`check_max_seq`, is used to check the maximum length of a sequence within a folder that contains .svg files.

`svg2xyList`, is used to convert a single .svg file into a list of x-y coordinates which refer to the path. This function requires the path and name of the .svg file as an input.
There are two variables 'clr' and 'clr1' which is suppose to represent the color of the lines. There are two because some .svg files specify the color attribute with 'STROKE' instead of 'stroke', and this capitalization caused the creation of an empty list. The parameter "epsilon" for rdp, used to vary the length of sequence is found in this function.

`to_stroke3`, is used to convert a list of x-y coordinates into stroke-3 format. The input needs to be a nested list of x-y coordinates.

`view_stroke3` is used to plot stroke-3 data.

`view_xylist` is used to plot the list of x-y coordinates.

There are 2 functions which convert .svg files into .npz, because it was used to check the difference between the function which converts x-y coordinate list into stroke-3 format.

`svg_to_npz` is the function which uses the stroke-3 conversion in this script, `to_stroke3`. `arr_reduce` has been implemented in the current version of this function.

`svg_to_npz_ex` is the function which uses `exp_w_order` to increase data by changing the order of strokes and reverse direction.
The input arguments of both functions are the same: 
'f_path' refers to the folder path which contains .svg files. 
't' is an integer which refers to the number of files that should be contained under the label 'train'. 
'v' is an integer which refers to the number of files that should be contained under the label 'valid'.
'tst' is an integer which refers to the number of files that should be contained under the label 'test'
'max_seq' is the variable used to specify what the maximum sequence length of the data should be. For example: when max_seq=250, only data which have less than or equal to a maximum sequence length of 250 will be compressed in the .npz file.

`svg_mix`, `svg_reverse` and `exp_w_order` works with `svg2xyList` properly. Needs to be modified to work with `rsvg_in _forlderxy`.

`svg_mix` used to create a list of sketch data with different order strokes. Takes input in the form of a nested list as produced by `svg2xyList`.

`svg_reverse` used to reverse the coordinate points in a line, with list for input.

`exp_w_order` meant to stand for expand with order. A function to expand svg data by changing order of strokes, and reversing strokes. The input for this function is a nested list in the form produced from `svg2xyList`.

`arr_reduce` is used to reduce the sequence length of data in stroke-3 format. Input for the function is 'data' which refers to data in stroke-3 format and 'lim' which refers to the preferred sequence length of data.

# npz_to_arr

This script is simply used to read and check the format of data in a .npz file.

# svg_mod

This script was used to compare the difference between data which includes white colored lines, and data which does not include white colored lines.
`w_wlines` reads an svg file and converts all lines into a list.
`wo_wlines` reads an svg file and converts only black lines into a list. It ignores white colored lines.
`flatten` is used to flatten nested lists into a single list. This was mainly used to check whether the content between two nested lists was the same or not. It was used to check the difference between stroke-3 conversion in `svg_parser` and `utils`.
Within this script is also lines which have been commented out but can be used to produce a histogram to show the distribution of sequence length of sketches. Instead of a histogram, a bar graph can also be shown to see the exact values.

# rem_invalid

This is used to remove sketches in sketchy database that are deemed invalid. Names of invalid files are written in invalid.txt of each folder.
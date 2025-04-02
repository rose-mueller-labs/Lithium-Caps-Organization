# from a list of filenames makes an Nx100x100 3d matrix where
    # N is the number of caps
    # 100x100 part is the grid of numbers
        # each number is the eggs for that smaller region and an associated secondary df with 
            # N rows where N is the number of images for that relevant metadeta

# the [ith] index into the image matrix is the [ith] row in the metadata df

# so i get a directory
# i go through the files
# in the file, i'm assuming its an image of the caps grid, let's say i have 30 caps
# so, it's going to be 30x100x100
# the 100x100 part is the GRID of NUMBESR where each NUMBER is the # of eggs for that smaller area
    # so that part is basically # of eggs in each 75x75 that makes up the bigger image
# the associated secondary df will have N number of rows which is the number of 75x75s at
# that spot i?
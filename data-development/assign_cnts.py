import os
import random

def assign_random_cnts(path):
    for filename in os.listdir(path):
        print(filename)
        old_file = os.path.join(path, filename)
        new_file = os.path.join(path, f'eggs{random.randint(1, 10)}count{filename}')
        os.rename(old_file, new_file)

if __name__ == '__main__':
    assign_random_cnts("/Users/shreyanakum/Downloads/Lithium-Caps-Organization/data-sliced")

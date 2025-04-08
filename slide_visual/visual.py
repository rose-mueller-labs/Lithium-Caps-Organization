# get the csv file with the predicted counts and actual and get the difference and graph it?
# step one:
# put the actual img tg
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import matplotlib.patches as patches

IMG_NAME = "ACO1 Control 04-30 22"

df = pd.read_csv("/home/drosophila-lab/Documents/Fecundity/Lithium-Caps-Organization/alex2.csv")

names = df['ImageName']

images_i_want = []

for i in names:
    if IMG_NAME in i:
        # print(i)
        images_i_want.append(i)

images_i_want = sorted(images_i_want)

#color = FF0000 #['red', 'cyan', 'purple', 'orange', 'green', 'yellow', 'blue', 'pink', 'brown', 'gray', 'olive']
colors = ['purple', 'violet', 'plum', 'thistle', 'mediumorchid', 'darkviolet', 'darkorchid', 'indigo']
#print('\n'.join(images_i_want))

ROOT_IMG = "/home/drosophila-lab/Documents/04-30-cap-800x800-sliced-Alexander/"

df_new = pd.read_csv('/home/drosophila-lab/Documents/Fecundity/Lithium-Caps-Organization/v1.csv')
fig, ax = plt.subplots(figsize=(6, 6))
for index, row in df_new.iterrows():
    if row['Filename'] in images_i_want:
        print(row['Filename'])
        full_path = os.path.join(ROOT_IMG, row['Filename'])

        # Load the image
        img = mpimg.imread(full_path)

        # Create a figure and axes
        x_coord = row['x']+0.5
        y_coord = 10-row['y']-0.5
        print(x_coord, y_coord)

        # Create an OffsetImage object with the image data
        imagebox = OffsetImage(img, zoom=0.45)  # Adjust zoom as needed

        # Create an AnnotationBbox to place the image at the specified coordinates
        ab = AnnotationBbox(imagebox, (x_coord, y_coord), frameon=False)
        print(ab)
        if (int(row['EggCount']) >= 0):
            # rect = patches.Rectangle((x_coord-0.5, y_coord+0.5-1), 1, 1, linewidth=2, edgecolor=f"#ff{int(row['EggCount'])*1000}", facecolor=f"#ff{int(row['EggCount'])*1000}", alpha=0.2)
            
            rect = patches.Rectangle((x_coord-0.5, y_coord+0.5-1), 1, 1, linewidth=2, edgecolor=colors[int(row['EggCount'])], facecolor=colors[int(row['EggCount'])], alpha=0.2)
            # Add the patch to the Axes
            ax.add_patch(rect)
            rect.set_zorder(4)

        ax.add_artist(ab)
        print(ax)

        ax.set_xlim(0, 10) 
        ax.set_ylim(0, 10)
        plt.xticks(np.arange(0, 10, step=1))
        plt.yticks(np.arange(0, 10, step=1))
        
        plt.plot()
    
    plt.show()

# Show the plot
plt.savefig("plot3")


# step two:
# apply the colors to it and reconstruct
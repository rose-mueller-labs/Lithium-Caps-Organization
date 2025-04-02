import os
import numpy as np
import time
import csv
import pandas as pd

# create a csv file with the predicted eggs and the actual
def create_csv_data_file(csv_name, data_path):
    with open(csv_name, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Population', 'Day', 'CapNumber','x', 'y', 'Filename', 'EggCount'])
        for img in os.listdir(f"{data_path}"):
            # eggs3countnCO1 Control 04-30 13 pt46.jpg
            # predicted_eggs = predict_egg_count(f"{data_path}/{label}/{img}")
            print(img)
            actual_eggs = img.split('eggs')[1].split('count')[0]
            population = ''.join(img.split('count')[1].split(' ')[0:2])
            day = img.split(' ')[2]
            cap_num = img.split(' ')[3]
            part = img.split('pt')[1].split('.')[0]
            x, y = get_x_y_coordinate_from_part(part)
            writer.writerow([population, day, cap_num, x, y, img, actual_eggs])

def get_actual_total(csv_path):
    # get all unique names => get the ones with the same names => get the actual counts => sum
    df = pd.read_csv(csv_path)
    root_image_names = np.array(df['RootImage'].unique())
    # print(root_image_names)
    actual_counts = dict()
    expected_counts = dict()
    for cap_name in root_image_names:
        actual_counts[cap_name] = 0
        expected_counts[cap_name] = 0

    for index, row in df.iterrows():
        actual_counts[row['RootImage']] += row['Actual']
        expected_counts[row['RootImage']] += row['Expected']

def get_x_y_coordinate_from_part(part):
    # part_number = int(part.split('pt')[1])
    part_number = int(part)

    if part_number % 10 == 0:
        return (int(part_number/10)-1, 10-1)
    x = int(np.floor(part_number / 10))
    y = (part_number % 10)-1

    return (x, y)


if __name__ == '__main__':
    # create_csv_data_file('alex2.csv', "/home/drosophila-lab/Documents/Fecundity/AlexanderDataClasses")
    # get_actual_total('alex2.csv', 'alex2_sums.csv')
    create_csv_data_file('v1.csv', "/home/drosophila-lab/Documents/04-30-cap-800x800-sliced-Alexander")
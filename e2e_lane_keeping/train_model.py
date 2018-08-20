import tensorflow as tf
import cnn_model
import scipy
import imageio
import csv

def getMetaDataFromFile(filename):
    data = []
    with open(filename, 'r') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')

        rowNum = 0
        for row in reader:

            if rowNum == 0:
                header = row
            else:
                data.append([row[0], row[3]])
            rowNum = rowNum + 1

    return data


if __name__ == '__main__':     
    data = getMetaDataFromFile('data/driving_log.csv')
    for d in data:
        print("Reading " + d[0])
        imageio.imread('data\\' + d[0])
        print(data)

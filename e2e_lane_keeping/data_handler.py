import scipy
import csv
import cv2

class DataHandler:

    def __init__(self, data_description_file):
        self.data_desc_file = data_description_file

        self.data = self.get_meta_data_from_file(self.data_desc_files)

    def get_meta_data_from_file(self, data_desc_file):
        data = []
        with open(data_desc_file, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')

            rowNum = 0
            for row in reader:

                if rowNum == 0:
                    header = row
                else:
                    image = self.get_image('data/' + row[0])
                    angle = row[3]
                    data.append([image, angle])
                rowNum = rowNum + 1

        return data

    def get_image(self, filename):
        image = scipy.misc.imresize(scipy.misc.imread(filename)[25:135], [66, 200])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        return (image / 255.0)

    def get_data_batch(self, batchSize, iteration):
        x_img = []
        y = []
        for d in self.data[batchSize * iteration:batchSize * iteration + batchSize]:
            x_img.append(d[0])
            y.append(d[1])

        return x_img, y

    def get_data_splits(self, val_split):
        if val_split < 1.0:
            train_data = self.data[0:int((1 - val_split) * len(self.data))]
            val_data = self.data[int((1 - val_split) * len(self.data)):]
        else:
            print("Invalid validation split. Split has to be < 1")

        return train_data, val_data


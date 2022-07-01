
from torch.utils.data import Dataset
import glob
import random
from PIL import Image

class Competition_No1_dataset(Dataset):
    def __init__(self, path, train=True, transform=None, index_shuf=[]):
        self.path = path

        if train:
            self.empty_path = path + '/train/0'
            self.loaded_path = path + '/train/1'
            self.empty_img_list = glob.glob(self.empty_path + '/*.png')
            self.loaded_img_list = glob.glob(self.loaded_path + '/*.png')
            self.transform = transform

            img_list_temp = self.empty_img_list + self.loaded_img_list
            class_list_temp = [0] * len(self.empty_img_list) + [1] * len(self.loaded_img_list)

            self.img_list = []
            self.class_list = []
            if (len(index_shuf) == 0):
                self.index_shuf = list(range(len(img_list_temp)))
                random.shuffle(self.index_shuf)
                for i in self.index_shuf:
                    self.img_list.append(img_list_temp[i])
                    self.class_list.append(class_list_temp[i])
            else:
                for i in index_shuf:
                    self.img_list.append(img_list_temp[i])
                    self.class_list.append(class_list_temp[i])

        else:
            self.empty_path = path + '/test/0'
            self.loaded_path = path + '/test/1'
            self.empty_img_list = glob.glob(self.empty_path + '/*.jpg')
            self.loaded_img_list = glob.glob(self.loaded_path + '/*.jpg')
            self.transform = transform

            self.img_list = self.empty_img_list + self.loaded_img_list
            self.class_list = [0] * len(self.empty_img_list) + [1] * len(self.loaded_img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img_path = self.img_list[idx]
        label = self.class_list[idx]

        #img = Image.open(img_path)  # RGB
        img = Image.open(img_path).convert("L") # gray

        if self.transform is not None:
            img = self.transform(img)

        sample = {'image': img, 'label': label, 'filename': img_path}

        return sample

    def get_index_shuf(self):
        return self.index_shuf
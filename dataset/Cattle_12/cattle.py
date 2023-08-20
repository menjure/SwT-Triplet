import os
from PIL import Image
from torch.utils.data import Dataset
import random

class cattle(Dataset):
    def __init__(self, root, train_val_test, transform = None, train_ratio_txt = ''):
      
        self.root = root
        self.train_val_test = train_val_test
        self.transform = transform
        self.train_ratio_txt = train_ratio_txt

        with open(os.path.join(self.root, self.train_ratio_txt), 'r') as f:
            dataset_split = f.readlines()

        self.train_files = []
        self.test_files = []

        self.train_label = []
        self.test_label = []
        
        if self.train_val_test == 'train':
            for dataline in dataset_split:
                id, image_path, label = dataline.strip().split()
                if label == '0':
                    self.train_files.append(image_path)
                    self.train_label.append(int(image_path.split('/')[0])-1)
        
        if self.train_val_test == 'test':
            for dataline in dataset_split:
                id, image_path, label = dataline.strip().split()
                if label == '1':
                    self.test_files.append(image_path)
                    self.test_label.append(int(image_path.split('/')[0])-1)


    def __len__(self):
        if self.train_val_test == 'train':
            return len(self.train_files)
        if self.train_val_test == 'test':
            return len(self.test_files)


    def __getitem__(self, index):
        if self.train_val_test == 'train':
            img_path = os.path.join(self.root, 'crop_nose_images', self.train_files[index]).replace('\\', '/')
            image_anchor = Image.open(img_path).convert('RGB')
            label_anchor = self.train_label[index]

            image_pos = self._retrievePositive(img_path)
            image_neg, label_neg = self._retrieveNegative(img_path)
            
            image_anchor = self.transform(image_anchor)
            image_pos = self.transform(image_pos)
            image_neg = self.transform(image_neg)

        if self.train_val_test == 'test':
            img_path = os.path.join(self.root, 'crop_nose_images', self.test_files[index]).replace('\\', '/')
            image_anchor = Image.open(img_path).convert('RGB')
            label_anchor = self.test_label[index]

            image_pos = self._retrievePositive(img_path)
            image_neg, label_neg = self._retrieveNegative(img_path)
            
            image_anchor = self.transform(image_anchor)
            image_pos = self.transform(image_pos)
            image_neg = self.transform(image_neg)

        return image_anchor, image_pos, image_neg, label_anchor, label_neg
        
    def getnumclass(self):
        directories = next(os.walk(os.path.join(self.root, 'crop_nose_images')))[1]
        return len(directories)

    
    def _retrievePositive(self, img_path):
        current_path = os.path.dirname(img_path) 
        file_list = os.listdir(current_path)

        if self.train_val_test == 'train':
            train = [i.split('/')[-1] for i in self.train_files]
            possible_list = [os.path.join(current_path, file).replace('\\', '/') for file in file_list if file in train]
        if self.train_val_test == 'test':
            test = [i.split('/')[-1] for i in self.test_files]
            possible_list = [os.path.join(current_path, file).replace('\\', '/') for file in file_list if file in test]
        
        assert img_path in possible_list
        possible_list.remove(img_path)

        pos_path = random.choice(possible_list)

        img = Image.open(pos_path).convert('RGB')
        return img

    def _retrieveNegative(self, img_path):
        possible_categories = os.listdir(os.path.dirname(os.path.dirname(img_path)))
        assert img_path.split('/')[-2] in possible_categories
        possible_categories.remove(img_path.split('/')[-2])

        random_category = random.choice(possible_categories)
        random_category = os.path.join(os.path.dirname(os.path.dirname(img_path)), random_category).replace('\\','/')

        file_list = os.listdir(random_category)

        if self.train_val_test == 'train':
            train = [i.split('/')[-1] for i in self.train_files]
            possible_list = [os.path.join(random_category, file).replace('\\', '/') for file in file_list if file in train]
        if self.train_val_test == 'test':
            test = [i.split('/')[-1] for i in self.test_files]
            possible_list = [os.path.join(random_category, file).replace('\\', '/') for file in file_list if file in test]
        
        neg_path = random.choice(possible_list)
        neg_label = int(neg_path.split('/')[-2])-1 

        img = Image.open(neg_path).convert('RGB')

        return img, neg_label

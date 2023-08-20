import os
import random
import platform
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

from models.embedding import SwT_embedding

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

        return image_anchor, image_pos, image_neg, label_anchor, label_neg, img_path
        
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

def getloader(data_root, batch_size, num_workers, train_ratio_txt):

    transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
    transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
    
    train_dataset = cattle(root=data_root, train_val_test='train', transform=transform_train, train_ratio_txt=train_ratio_txt)
    train_dataloder = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers)
    
    test_dataset = cattle(root=data_root,train_val_test='test', transform=transform_test, train_ratio_txt=train_ratio_txt)
    test_dataloder = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers)

    return train_dataloder, test_dataloder, train_dataset, test_dataset

def test(args):

    if platform.system() == 'Windows':
        num_workers = 0  
    else:

        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  

    train_dataloder, test_dataloder, train_dataset, test_dataset = getloader(args.data_root, 
                                                            args.batch_size, num_workers, args.train_ratio_txt)
    neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-4)
    assert train_dataset.getnumclass() == test_dataset.getnumclass()

    
    model = SwT_embedding(num_classes=train_dataset.getnumclass(), model_path=args.model_path, 
                          embedding_size=args.embedding_size)
    model.cuda()
    model.eval()
    
    # train inferEmbeddings
    train_embeddings = np.zeros((1,args.embedding_size))
    train_labels = np.zeros((1))
    for images, _, _, labels, _, img_path in tqdm(train_dataloder, desc=f"Inferring train embeddings"):
        images = Variable(images.cuda())
        outputs = model(images)
        embeddings = outputs.data    
        embeddings = embeddings.cpu().numpy()
        labels = labels.view(len(labels))
        labels = labels.cpu().numpy()
        train_embeddings = np.concatenate((train_embeddings,embeddings), axis=0)
        train_labels = np.concatenate((train_labels,labels), axis=0)
    train_embeddings = train_embeddings[1:]  
    train_labels = train_labels[1:]

    if args.save_embeddings:
        os.makedirs(args.save_path, exist_ok=True)
		# Construct the save path
        save_path = os.path.join(args.save_path, "train_embeddings.npz").replace('\\', '/')
		
		# Save the embeddings to a numpy array
        np.savez(save_path, embeddings=train_embeddings, labels=train_labels)
        print('save embeddings done!')
    
    neigh.fit(train_embeddings, train_labels)

    # test inferEmbeddings
    correct = 0
    total = 0
    error = {}
    for images, _, _, labels, _, img_path in tqdm(test_dataloder, desc=f"Inferring test embeddings"):
        images = Variable(images.cuda())
        outputs = model(images)
        embeddings = outputs.data    
        embeddings = embeddings.cpu().numpy()
        labels = labels.view(len(labels))
        labels = labels.cpu().numpy()

        predict = neigh.predict(embeddings)
        if predict == labels:
            correct += 1
        # else:
        #     img_path = str(img_path).split("'")[1].split("images/")[1]
        #     error[img_path] = f"图片{img_path}预测ID为{int(predict+1)}, 真实ID为{int(labels+1)}"
        total += 1

    accuracy = float((correct / total) * 100)

    return accuracy, error


def main():
    args = parser()

    accuracy, error = test(args)
    print(f'accuracy: {accuracy:.3f}%')
    for i in error:
        print(error[i])



def parser():
    parser = argparse.ArgumentParser(description="Params")

    parser.add_argument("--model_path", type=str, default='weights/best_acc_weights.pth')

    parser.add_argument("--data_root", type=str, default='datasets/Cattle_12')

    parser.add_argument("--train_ratio_txt", type=str, default='dataset_split.txt')

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument('--embedding_size', type=int, default=128)

    parser.add_argument('--save_path', type=str, default='embeddings',
						help="Where to store the embeddings")
    
    parser.add_argument('--save_embeddings', type=bool, default=True,
						help="Should we save the embeddings to file")
    

    return parser.parse_args()


if __name__ == '__main__':
    main()
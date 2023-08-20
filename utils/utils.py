import os
import logging
import platform
import numpy as np
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.Cattle_12.cattle import cattle
from models.embedding import SwT_embedding


def getLogger(filename):
    logger = logging.getLogger('train_logger')

    while logger.handlers:
        logger.handlers.pop()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, 'w')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s], ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


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


def train(model, train_dataloder, train_dataset, device, loss_function, optimizer):
    print('training...')

    model.train()
    train_loss1=0.0
    train_loss2=0.0
    train_loss3=0.0
    train_loss4=0.0
    train_loss5=0.0

    for step, (img_anchor, img_pos, img_neg, label_anchor, label_neg) in enumerate(train_dataloder):
        img_anchor = Variable(img_anchor.to(device))
        img_pos = Variable(img_pos.to(device))
        img_neg = Variable(img_neg.to(device))

        optimizer.zero_grad()
        embed_anch, embed_pos, embed_neg, preds = model(img_anchor, img_pos, img_neg)

        loss, triplet_loss, softmax_loss, ap_distances, an_distances = loss_function(embed_anch, embed_pos, embed_neg, preds, label_anchor, label_neg)

        loss.backward()
        optimizer.step()

        train_loss1 += loss.item()
        train_loss2 += triplet_loss.item()
        train_loss3 += softmax_loss.item()
        train_loss4 += ap_distances.item()
        train_loss5 += an_distances.item()
            
        if (step+1) % (int(len(train_dataloder)/4)) == 0:
            print(f"step: {step+1}/{len(train_dataloder)}, loss: {loss:.4f}, "
                  f"triplet_loss: {triplet_loss:.4f}, softmax_loss: {softmax_loss:.4f} ap_distances: {ap_distances:.3f} an_distances: {an_distances:.3f}")

    train_loss1 = train_loss1/len(train_dataloder)
    train_loss2 = train_loss2/len(train_dataloder)
    train_loss3 = train_loss3/len(train_dataloder)
    train_loss4 = train_loss4/len(train_dataloder)
    train_loss5 = train_loss5/len(train_dataloder)

    return train_loss1, train_loss2, train_loss3, train_loss4, train_loss5


def val(model, test_dataloder, test_dataset, device, loss_function):
    print('valid...')

    model.eval()
    test_acc = 0.0
    test_loss = 0.0

    with torch.no_grad():
        for _, (test_images, test_labels) in enumerate(test_dataloder):
            test_images = Variable(test_images.to(device))
            test_labels = Variable(test_labels.to(device))
            test_output = model(test_images)

            v_loss = loss_function(test_output, test_labels)
            test_loss += v_loss.item()
            test_predicted = torch.max(test_output, dim=1)[1]
            test_acc += (test_predicted==test_labels).sum().item()
            
        test_accurate = 100*test_acc/len(test_dataset)
        test_loss = test_loss/len(test_dataloder)

        return test_accurate, test_loss


# Save a checkpoint as the current state of training
def saveCheckpoint(save_folder, epoch, model, optimiser, description):
    # Construct a state dictionary for the training's current state
    state = {'epoch': epoch,
             'model_state': model.state_dict(),
             'optimizer_state': optimiser.state_dict()}

    # Construct the full path for where to save this
    checkpoint_path = os.path.join(save_folder, f"{description}_model_state.pth").replace('\\', '/')

    # And save actually it
    torch.save(state, checkpoint_path)

    return checkpoint_path


def test(args, checkpoint_path):


    if platform.system() == 'Windows':
        num_workers = 0  
    else:
        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  

    train_dataloder, test_dataloder, train_dataset, test_dataset = getloader(args.data_root, 
                                                            args.batch_size, num_workers, args.train_ratio_txt)

    train_embeddings, train_labels = inferEmbeddings(args, train_dataloder, train_dataset, "train", checkpoint_path)
    test_embeddings, test_labels = inferEmbeddings(args, test_dataloder, test_dataset, "test", checkpoint_path)

    # Classify them
    accuracy = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)

    return accuracy


# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    total = len(test_labels)

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)

    # How many were correct?
    correct = (predictions == test_labels).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100

    return accuracy


# Infer the embeddings for a given dataset
def inferEmbeddings(args, dataloder, dataset, split, checkpoint_path):
    if platform.system() == 'Windows':
        num_workers = 0  
    else:
        # 在 Linux 系统下设置为 CPU 核心数
        num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  

	# Define our embeddings model
    model = SwT_embedding(num_classes=dataset.getnumclass(), model_path=checkpoint_path, embedding_size=args.embedding_size)
	
	# Put the model on the GPU and in evaluation mode
    model.cuda()
    model.eval()

	# Embeddings/labels to be stored on the testing set
    outputs_embedding = np.zeros((1,args.embedding_size))
    labels_embedding = np.zeros((1))
    total = 0
    correct = 0

	# Iterate through the testing portion of the dataset and get
    for images, _, _, labels, _ in tqdm(dataloder, desc=f"Inferring {split} embeddings"):
        # Put the images on the GPU and express them as PyTorch variables
        images = Variable(images.cuda())

        # Get the embeddings of this batch of images
        outputs = model(images)

        # Express embeddings in numpy form
        embeddings = outputs.data    
        embeddings = embeddings.cpu().numpy()

        # Convert labels to readable numpy form
        labels = labels.view(len(labels))
        labels = labels.cpu().numpy()

        # Store testing data on this batch ready to be evaluated
        outputs_embedding = np.concatenate((outputs_embedding,embeddings), axis=0)
        labels_embedding = np.concatenate((labels_embedding,labels), axis=0)
	
    outputs_embedding = outputs_embedding[1:]  
    labels_embedding = labels_embedding[1:]

	# If we're supposed to be saving the embeddings and labels to file
    if args.save_embeddings:
        os.makedirs(args.save_path, exist_ok=True)
		# Construct the save path
        save_path = os.path.join(args.save_path, f"{split}_embeddings.npz").replace('\\', '/')
		
		# Save the embeddings to a numpy array
        np.savez(save_path, embeddings=outputs_embedding, labels=labels_embedding)
        print('save embeddings done!')

    return outputs_embedding, labels_embedding


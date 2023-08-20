import torch
import torch.nn as nn
from torch.autograd import Variable


class OnlineReciprocalSoftmaxLoss(nn.Module):
	def __init__(self, triplet_selector,  lambda_factor=0.01):
		super(OnlineReciprocalSoftmaxLoss, self).__init__()
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Define the labels as variables and put on the GPU
		gpu_labels = labels.view(len(labels))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		gpu_labels = Variable(gpu_labels.cuda())
		gpu_labels_neg = Variable(gpu_labels_neg.cuda())

		# Concatenate labels for softmax/crossentropy targets
		target = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1).sqrt()
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1).sqrt()
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1).sqrt()
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1).sqrt()
		
		# Compute the triplet losses
		triplet_losses = ap_distances + (1/an_distances)

		# Compute softmax loss		
		loss_softmax = self.loss_fn(input=preds, target=target)

		# Compute the total loss
		loss_total = self.lambda_factor*triplet_losses.mean() + loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax, ap_distances.mean(), an_distances.mean()

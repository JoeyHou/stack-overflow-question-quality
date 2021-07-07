# Reference Code:
# https://github.com/aws-samples/amazon-sagemaker-bert-pytorch/blob/master/code/train_deploy.py

######### Imports #########
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

######### Important Variables #########
# max length of the sentence
MAX_LEN = 128 

# BERT Tokenizer
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# def flat_accuracy(preds, labels):
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return np.sum(pred_flat == labels_flat) / len(labels_flat)

######### DataLoader Functions #########
def _get_train_data_loader(batch_size, training_dir, validation, is_distributed = False):
    
    logger.info("Getting train dataloader!")
    
    # 1. Load data
    if validation:
        dataset = pd.read_csv(os.path.join(training_dir, "val_s3.csv"))
    else:
        dataset = pd.read_csv(os.path.join(training_dir, "train_s3.csv"))
    sentences = dataset.sentence.values
    labels = dataset.label.values
    
    # 2. Encode text
    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        input_ids.append(encoded_sent)
    
    # 3. Pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded
    
    # 4. Adding mask; mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    
    # 5. Convert to PyTorch data types.
    train_inputs = torch.tensor(input_ids)
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader

def _get_test_data_loader(batch_size, training_dir):
    logger.info("Getting test dataloader!")
    
    # 1. Load data
    dataset = pd.read_csv(os.path.join(training_dir, "test_s3.csv"))
    sentences = dataset.sentence.values
    labels = dataset.label.values
    
    # 2. Encode text
    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
        input_ids.append(encoded_sent)
        
    # 3. Pad shorter sentences
    input_ids_padded = []
    for i in input_ids:
        while len(i) < MAX_LEN:
            i.append(0)
        input_ids_padded.append(i)
    input_ids = input_ids_padded

    # mask; 0: added, 1: otherwise
    attention_masks = []
    # For each sentence...
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # convert to PyTorch data types.
    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return test_dataloader

######### Training Functions #########
def train(args):
    
    # 1. Device settings and distributed computing status
    #is_distributed = len(args.hosts) > 1 and args.backend is not None
    is_distributed = False
    logger.debug("Distributed training - %s", is_distributed)

    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - %d", args.num_gpus)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # 2. Load train&test dataloader
    # TODO1: dataloader functions
    ## --- Your code here --- ##
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, validation = False)
    val_loader = _get_train_data_loader(args.batch_size, args.data_dir, validation = True)
    test_loader = _get_test_data_loader(args.test_batch_size, args.test)
    ## --- Your code ends --- ##
    
    # 3. Model definition
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # 4. Training settings (optimizer, distributed computing, etc.)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8,  # args.adam_epsilon - default is 1e-8.
    )

    # 5. Trains the model
    all_epoches = []
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        model.train()
        for step, batch in tqdm(enumerate(train_loader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            if step % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )
        logger.info("Average training loss: %f\n", total_loss / len(train_loader))
        
        train_acc = test(model, train_loader, device)
        val_acc = test(model, val_loader, device)
        logger.info("Train accuracy: %f\n", train_acc)
        logger.info("Val accuracy: %f\n", val_acc)
                
        all_epoches.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'model': model
        })
        
    # 6. Pick the best model to test
    best_model = all_epoches.sort_values(by = 'val_acc', ascending = False).model.values[0]
    logger.info("Test accuracy: %f\n", test(best_model, test_loader, device))
    
    # 7. Save the best model
    logger.info("Saving tuned model.")
    model_2_save = best_model.module if hasattr(best_model, "module") else best_model
    model_2_save.save_pretrained(save_directory=args.model_dir)  

def test(model, test_loader, device):
    model.eval()
    eval_accuracy = 0
    
    all_pred = []
    all_label = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            
            for pred in np.argmax(logits, axis=1).flatten():
                all_pred.append(pred)
            for label in label_ids.flatten():
                all_label.append(label)
            #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            #eval_accuracy += tmp_eval_accuracy
    all_pred = np.array(all_pred)
    all_label = np.array(all_label)
    eval_accuracy = np.sum(all_pred == all_label) / len(all_label)
    logger.info("Test set: Accuracy: %f\n", eval_accuracy)
    return eval_accuracy


# Main function
if __name__ == '__main__':

    # Trainer #1
#     # All of the model parameters and training parameters are sent as arguments
#     # when this script is executed, during a training job
    
#     # Here we set up an argument parser to easily access the parameters
#     parser = argparse.ArgumentParser()

#     # SageMaker parameters, like the directories for training data and saving models; set automatically
#     # Do not need to change
#     # parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
#     parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#     parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    
#     # Model Parameters
#     parser.add_argument('--num_labels', type=int, default=3, metavar='N',
#                         help='number of labels for the dataset (default: 3)')

#     # Training Parameters
#     parser.add_argument(
#         "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
#     )
#     parser.add_argument(
#         "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
#     )
#     parser.add_argument("--epochs", type=int, default=3, metavar="N", help="number of epochs to train (default: 10)")
#     parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
#     parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
#     parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
#     parser.add_argument(
#         "--log-interval",
#         type=int,
#         default=50,
#         metavar="N",
#         help="how many batches to wait before logging training status",
#     )
#     parser.add_argument(
#         "--backend",
#         type=str,
#         default=None,
#         help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
#     )

#     # Container environment
#     parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
#     parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
#     #parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
#     #parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
#     parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
#     parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    
#     # args holds all passed-in arguments
#     args = parser.parse_args()
#     train(args)

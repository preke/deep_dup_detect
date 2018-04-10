import os
import sys
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def train(train_iter, vali_iter, model, args):
    if args.cuda:
        model.cuda()
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))    
    optimizer  = torch.optim.Adam(parameters, lr=args.lr)
    steps      = 0
    last_step  = 0
    
    model.train()
    log_file = open('log.txt', 'w')
    for epoch in range(1, args.epochs+1):
        print('\nEpoch:%s\n'%epoch)
        log_file.write('\nEpoch:%s\n'%epoch)
        for batch in train_iter:
            question1, question2, label = batch.question1, batch.question2, batch.label
            label.data.sub_(1)
            if args.cuda:
                question1, question2, label = question1.cuda(), question2.cuda(), label.cuda()
            optimizer.zero_grad()
            logit = model(question1, question2)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            steps += 1
            

            if steps % args.log_interval == 0:
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}'.format(steps, loss.data[0]))
                log_file.write('\rBatch[{}] - loss: {:.6f}'.format(steps, loss.data[0]))
                length = len(target.data)
                for i in range(length):
                    a = logit.data[i]
                    b = target.data[i]
                corrects = 0
                if a <= 0 and b == 0:
                    corrects += 1
                elif a > 0 and b == 1:
                    corrects += 1
                accuracy = 100.0 * corrects/batch.batch_size
                print('\n acc: %s'%str(accuracy))
            


            if steps % args.test_interval == 0:
                vali_loss = eval(vali_iter, model, args).data[0]
                if vali_loss < min_loss:
                    min_loss = vali_loss
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                elif steps - last_step >= args.early_stop:
                    print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                print('save loss: %s' %str(loss.data))
                log_file.write('save loss: %s\n' %str(loss.data))
                save(model, args.save_dir, 'snapshot', steps)

    log_file.close()

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

def eval(vali_iter, model, args):
    model.eval()
    accuracy, avg_loss = 0, 0
    for batch in vali_iter:
        question1, question2, label = batch.question1, batch.question2, batch.label
        label.data.sub_(1)
        if args.cuda:
            question1, question2, label = question1.cuda(), question2.cuda(), label.cuda()
        logit = model(question1, question2)
        loss = F.cross_entropy(logit, target)
        return loss

def test(test_iter, model, args):
    accuracy = 0.0
    total_num = 0.0
    for batch in test_iter:
        question1, question2, label = batch.question1, batch.question2, batch.label
        label.data.sub_(1)
        if args.cuda:
            question1, question2, label = question1.cuda(), question2.cuda(), label.cuda()

        logit = model(question1, question2)
        length = len(target.data)
        for i in range(length):
            a = logit.data[i]
            b = target.data[i]
            corrects = 0
            if a <= 0 and b == 0:
                corrects += 1
            elif a > 0 and b == 1:
                corrects += 1
        accuracy = 100.0 * corrects/batch.batch_size
        print('\n test acc: %s'%str(accuracy))





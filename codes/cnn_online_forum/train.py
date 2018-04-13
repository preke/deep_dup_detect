# coding = utf-8
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import traceback
def train(train_iter, vali_iter, model, args):
    if args.cuda:
        model.cuda()
    # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    
    # epoch 是 训练的 round
    
    for epoch in range(1, args.epochs+1): 
        print('\nEpoch:%s\n'%epoch)
        
        model.train()
        for batch in train_iter:
            question1, question2, target = batch.question1, batch.question2, batch.label
            if args.cuda:
                question1, question2, target = question1.cuda(), question2.cuda(), target.cuda()
            optimizer.zero_grad()
            logit = model(question1, question2)
            target = target.type(torch.cuda.FloatTensor)
            criterion = nn.MSELoss()
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()
            
            '''
                记录每次model返回一个pair的相似度
                手动计算MSE
            '''
            

            steps += 1
            if steps % args.log_interval == 0:
                corrects = 0 
                length = len(target.data)
                for i in range(length):
                    a = logit.data[i]
                    b = target.data[i]
                    if a < 0.6 and b == 0:
                        corrects += 1
                    elif a >= 0.6 and b == 1:
                        corrects += 1
                    else:
                        pass
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                vali_acc = eval(vali_iter, model, args)
                if vali_acc > best_acc:
                    best_acc = vali_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                #
            elif steps % args.save_interval == 0:
                print('save loss: %s' %str(loss.data))
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        question1, question2, target = batch.question1, batch.question2, batch.label
        if args.cuda:
            question1, question2, target = question1.cuda(), question2.cuda(), target.cuda()

        logit = model(question1, question2)
        target = target.type(torch.cuda.FloatTensor)
        criterion = nn.MSELoss()
        length = len(target.data)
        for i in range(length):
            a = logit.data[i]
            b = target.data[i]
            if a < 0.6 and b == 0:
                corrects += 1
            elif a >= 0.6 and b == 1:
                corrects += 1
            else:
                pass
        
    size = float(len(data_iter.dataset))
    avg_loss /= size
    accuracy = 100.0 * float(corrects)/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def test(test_iter, model, args):
    accuracy = 0.0
    total_num = 0.0
    threshold = 0.6
    for batch in test_iter:
        question1, question2, label = batch.question1, batch.question2, batch.label
        if args.cuda:
            question1, question2, label = question1.cuda(), question2.cuda(), label.cuda()
        label = label.type(torch.cuda.FloatTensor)   
        results = model(question1, question2)
        for i in range(len(label.data)):
            if (label.data[i] == 1) and (results.data[i] >= threshold):
                accuracy += 1.0
            elif (label.data[i] == 0) and (results.data[i] < threshold):
                accuracy += 1.0
            else:
                pass
            
        total_num += len(label.data)
    # print(accuracy)
    # print(total_num)
    print('Threshold is: %s, Accuracy is: %s' %(str(threshold), str(accuracy/total_num)))


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

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
    min_loss   = 10000
    model.train()
    log_file = open('log.txt', 'w')
    for epoch in range(1, args.epochs+1):
        print('\nEpoch:%s\n'%epoch)
        log_file.write('\nEpoch:%s\n'%epoch)
        for batch in train_iter:
            query, pos_doc, neg_doc_1, neg_doc_2, neg_doc_3, neg_doc_4, neg_doc_5 = \
            batch.query, batch.pos_doc, batch.neg_doc_1, batch.neg_doc_2, batch.neg_doc_3, batch.neg_doc_4, batch.neg_doc_5
            # query.t_(), pos_doc.t_(), neg_doc_1.t_(), neg_doc_2.t_(), neg_doc_3.t_(), neg_doc_4.t_(), neg_doc_5.t_()
            if args.cuda:
                query, pos_doc, neg_doc_1, neg_doc_2, neg_doc_3, neg_doc_4, neg_doc_5 = \
                query.cuda(), pos_doc.cuda(), neg_doc_1.cuda(), neg_doc_2.cuda(), neg_doc_3.cuda(), neg_doc_4.cuda(), neg_doc_5.cuda()
            
            optimizer.zero_grad()
            results = torch.cat([model(query, pos_doc).view(-1,1), model(query, neg_doc_1).view(-1,1)], 1)
            results = torch.cat([results, model(query, neg_doc_2).view(-1,1)], 1)
            results = torch.cat([results, model(query, neg_doc_3).view(-1,1)], 1)
            results = torch.cat([results, model(query, neg_doc_4).view(-1,1)], 1)
            results = torch.cat([results, model(query, neg_doc_5).view(-1,1)], 1)
            criterion  = nn.NLLLoss()
            target_tmp = Variable(torch.LongTensor(np.array([0], dtype=float)))
            target     = target_tmp
            for i in range(results.shape[0] - 1):
                target = torch.cat([target, target_tmp])
            if args.cuda:
                target = target.cuda()
            log_softmax = nn.LogSoftmax(dim = 1)
            loss = criterion(log_softmax(results), target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                sys.stdout.write('\rBatch[{}] - loss: {:.6f}'.format(steps, loss.data[0]))
                log_file.write('\rBatch[{}] - loss: {:.6f}'.format(steps, loss.data[0]))
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
        query, pos_doc, neg_doc_1, neg_doc_2, neg_doc_3, neg_doc_4, neg_doc_5 = \
            batch.query, batch.pos_doc, batch.neg_doc_1, batch.neg_doc_2, batch.neg_doc_3, batch.neg_doc_4, batch.neg_doc_5
        if args.cuda:
            query, pos_doc, neg_doc_1, neg_doc_2, neg_doc_3, neg_doc_4, neg_doc_5 = \
                query.cuda(), pos_doc.cuda(), neg_doc_1.cuda(), neg_doc_2.cuda(), neg_doc_3.cuda(), neg_doc_4.cuda(), neg_doc_5.cuda()

        results = torch.cat([model(query, pos_doc).view(-1,1), model(query, neg_doc_1).view(-1,1)], 1)
        results = torch.cat([results, model(query, neg_doc_2).view(-1,1)], 1)
        results = torch.cat([results, model(query, neg_doc_3).view(-1,1)], 1)
        results = torch.cat([results, model(query, neg_doc_4).view(-1,1)], 1)
        results = torch.cat([results, model(query, neg_doc_5).view(-1,1)], 1)
        criterion  = nn.NLLLoss()
        target_tmp = Variable(torch.LongTensor(np.array([0], dtype=float)))
        target     = target_tmp
        for i in range(results.shape[0] - 1):
            target = torch.cat([target, target_tmp])
        if args.cuda:
            target = target.cuda()
        log_softmax = nn.LogSoftmax(dim = 1)
        loss = criterion(log_softmax(results), target)
        return loss

def test(test_iter, model, args):
    accuracy = 0.0
    total_num = 0.0
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        for batch in test_iter:
            query, doc, label = batch.query, batch.doc, batch.label
            if args.cuda:
                query, doc, label = query.cuda(), doc.cuda(), label.cuda()

            results = model(query, doc)
            for i in range(len(label.data)):
                # print('label:%s\n' %str(label.data[i]))
                # print('results:%s\n' %str(results.data[i]))

                if (label.data[i] == 1) and (results.data[i] >= threshold):
                    accuracy += 1.0
                elif (label.data[i] == 2) and (results.data[i] < threshold):
                    accuracy += 1.0
                else:
                    pass
            
            total_num += len(label.data)
        # print(accuracy)
        # print(total_num)
        print('Threshold is: %s, Accuracy is: %s' %(str(threshold), str(accuracy/total_num)))




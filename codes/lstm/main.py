'''
    考虑如何解决两句话很相似，但是仅仅因为一两个字不同导致不是一个label为重复的pair?
    损失函数 or 相似度函数 由两部分组成：
      1. lstm(bilstm) 的hidden layer
      2. bag of words 的 差集
      调权重
'''
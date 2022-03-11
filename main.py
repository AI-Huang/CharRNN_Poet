#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-04-22 17:56
# @Author  : Kelley Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/chenyuntc/pytorch-book/blob/master/chapter09-neural_poet_RNN/main.py

import sys
import os
import torch
from data import get_data
from model import PoetryModel
from torch import nn
from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb

from config import Config
opt = Config()
opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：
    """
    device = opt.device

    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(device)

    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)

        if w == '<EOP>':
            del results[-1]
            break

    return results


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    """
    device = opt.device

    results = []
    start_word_len = len(start_words)
    input = (torch.Tensor([word2ix['<START>']]).view(1, 1).long()).to(device)

    hidden = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = '<START>'

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w

    return results


def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """
    device = opt.device

    for k, v in kwargs.items():
        setattr(opt, k, v)
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256).to(device)
    def map_location(s, l): return s
    state_dict = torch.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.to(device)

    # python2 和 python3 字符串兼容
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode(
                'ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode(
            'utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = opt.device

    vis = Visualizer(env=opt.env)

    # 获取数据
    data, word2ix, ix2word = get_data(opt)
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), 128, 256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))

    loss_meter = meter.AverageValueMeter()
    for epoch in range(opt.epoch):
        loss_meter.reset()

        for i, data_ in tqdm.tqdm(enumerate(dataloader)):
            # 训练
            optimizer.zero_grad()

            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            # 可视化
            if i % opt.plot_every == 0:

                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])

                # 当前输入批次的诗歌原文
                poetrys = [
                    [ix2word[_word] for _word in data_[:, _].tolist()]
                    for _ in range(data_.shape[1])
                ][:16]  # 取前16首诗歌
                vis.text('</br>'.join([''.join(poetry)
                                       for poetry in poetrys]), win=u'origin_poem')

                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u"春江花月夜凉如水"):
                    gen_poetry = ''.join(
                        generate(model, word, ix2word, word2ix))
                    gen_poetries.append(gen_poetry)
                vis.text('</br>'.join([''.join(poetry)
                                       for poetry in gen_poetries]), win=u'gen_poem')

        torch.save(model.state_dict(),
                   f'{opt.model_ckpt_dir}_{str(epoch)}.pth')


if __name__ == '__main__':
    import fire

    fire.Fire()

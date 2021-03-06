#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Mar-08-22 19:30
# @Author  : Kelley Kan HUANG (kan.huang@connect.ust.hk)

class Config(object):
    data_path = 'data/'  # 诗歌的文本文件存放路径
    pickle_path = 'data/processed/tang.npz'  # 预处理好的二进制文件
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3
    weight_decay = 1e-4
    epoch = 20
    batch_size = 32  # 128
    maxlen = 125  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格

    use_gpu = True  # 是否使用GPU

    plot_every = 20  # 每20个batch可视化一次
    use_env = True  # 是否使用visodm
    env = 'poetry'  # visdom env

    max_gen_len = 200  # 生成诗歌最长长度
    debug_file = '/tmp/debugp'
    model_path = None  # 预训练模型路径
    prefix_words = '细雨鱼儿出,微风燕子斜。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '闲云潭影日悠悠'  # 诗歌开始
    acrostic = False  # 是否是藏头诗
    model_ckpt_dir = 'checkpoints/tang'  # 模型保存路径

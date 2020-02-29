# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import visdom
import torchvision
from memory_profiler import profile # 内存占用分析插件

viz = visdom.Visdom()
debug = True

from codes.utils import get_subwindow_tracking, get_search_region_target, crop_image

def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1

# 计算目标的得分图
# @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
def tracker_eval(im, avg_chans, net, x_crop, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop)
    # print("方差:", score.squeeze(0).mean(0).mean().item() / score.squeeze(0).mean(0).var().item())
    # 实时显示得分的变化
    # background_score = score[:, :5, :, :].squeeze()
    # target_score = score[:, 5:, :, :].squeeze()
    # diff_score = (target_score - background_score)
    # viz.heatmap(diff_score.mean(0), opts=dict(title='diff_score',
    #                                           caption='diff_score.'), win="diff_score")
    # 用于边框回归的量,表示的是分别对5中anchor进行回归, 其结构展开来应该是(4, 5, 19, 19)
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    # 目标打分的量, 表示的是对5种anchor分别处理下的打分的量,其结构展开来应该为(5, 19, 19)
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    b = pscore.reshape(5, 19, 19)
    # b = np.concatenate(b)
    viz.heatmap(b.mean(0), win="Pscore", opts={"title":"Pscore"})
    # 返回前4个最大值的索引
    best_pscore_id = np.argmax(pscore)

    # 计算得到k个得分最高样本的尺寸和位置
    if (len(net.memory.manager.support_set_list) >= net.memory.store_amount) and debug:
        # best_pscore_id = np.argmax(pscore)
        k = 1000
        best_pscore_id = torch.from_numpy(pscore).topk(k)
        # 按间隔选取, 以免同一位置处的元素过多
        select_index = torch.arange(start=0, end=k, step=k // 20)
        best_pscore_id = best_pscore_id[1][select_index]

        target_szs = []
        target_poss = []
        confidences = []
        target_sz_cp = target_sz.copy()
        target_pos_cp = target_pos.copy()
        for i in range(best_pscore_id.size(0)):
            target = delta[:, best_pscore_id[i]] / scale_z
            target_sz = target_sz_cp / scale_z
            lr = penalty[best_pscore_id[i]] * score[best_pscore_id[i]] * p.lr

            res_x = target[0] + target_pos_cp[0]
            res_y = target[1] + target_pos_cp[1]
            res_w = target_sz[0] * (1 - lr) + target[2] * lr
            res_h = target_sz[1] * (1 - lr) + target[3] * lr
            target_pos = np.array([res_x, res_y])
            target_sz = np.array([res_w, res_h])

            target_szs.append(target_sz)
            target_poss.append(target_pos)

            rect = np.concatenate([target_pos - target_sz // 2, target_sz])
            z_crop_candidate = crop_image(im, rect, img_size=127, padding=10).unsqueeze(0)
            # wc_z = target_sz[0] + p.context_amount * sum(target_sz)
            # hc_z = target_sz[1] + p.context_amount * sum(target_sz)
            # s_z = round(np.sqrt(wc_z * hc_z))
            # z_crop_candidate = get_subwindow_tracking(im, target_pos, p.exemplar_size,
            #                                 s_z, avg_chans)

            z_crop_candidate = net.featureExtract(z_crop_candidate).view(1,-1)
            # z_crop_candidate = net.featureExtract(crop_image(im, rect, padding=10).unsqueeze(0))
            confidences.append(net.memory.manager.measure_d.forward(net.memory.support_set, z_crop_candidate).max().item())
            # viz.image(crop_image(im, rect, padding=10), opts={"title":"Score:{}".format(confidences[i])})
        viz.line(Y=[np.mean(confidences)], X=[net.step], update='append', win='confidence')
        net.step += 1

        top_trust = np.argmax(confidences)
        target_pos = target_poss[top_trust]
        target_sz = target_szs[top_trust]
        best_pscore_id = best_pscore_id[top_trust]
    else:
        target = delta[:, best_pscore_id] / scale_z
        target_sz = target_sz / scale_z
        lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])

    # 更新支撑样本集合
    rect = np.concatenate([target_pos - target_sz // 2, target_sz])
    gt = crop_image(im, rect, img_size=127, padding=5).unsqueeze(0)
    # wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    # hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    # s_z = round(np.sqrt(wc_z * hc_z))
    # gt = get_subwindow_tracking(im, target_pos, p.exemplar_size,
    #                                           s_z, avg_chans)
    viz.image(gt.squeeze(), win="Current Result")

    z_crop_gt = net.featureExtract(gt).view(1,-1)
    net.memory.insert_support_gt(z_crop_gt, gt)

    return target_pos, target_sz, score[best_pscore_id]

# 初始化跟踪器网络
def SiamRPN_init(im, target_pos_init, target_sz_init, net):
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if p.adaptive:
        if ((target_sz_init[0] * target_sz_init[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))
    # 计算图像3个通道的平均值
    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz_init[0] + p.context_amount * sum(target_sz_init)
    hc_z = target_sz_init[1] + p.context_amount * sum(target_sz_init)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos_init, p.exemplar_size,
                                    s_z, avg_chans)
    z = Variable(z_crop.unsqueeze(0))

    # 将第一帧的目标送入到内容管理器中, 以监督内容管理器的质量
    rect = np.concatenate([target_pos_init - target_sz_init // 2, target_sz_init])
    z_crop_candidate = crop_image(im, rect, img_size=127, padding=10).unsqueeze(0)

    net.memory.insert_init_gt(net.featureExtract(z_crop_candidate))

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['target_pos'] = target_pos_init
    state['target_sz'] = target_sz_init

    state['exemplar_size'] = p.exemplar_size
    state['s_z'] = s_z

    # 传入目标的位置和大小, 还有第一帧的模板
    net.temple(z)

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)
    state['window'] = window

    return state

# 跟踪
# @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
def SiamRPN_track(state, im):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos_old = state['target_pos']
    target_sz_old = state['target_sz']

    wc_z = target_sz_old[1] + p.context_amount * sum(target_sz_old)
    hc_z = target_sz_old[0] + p.context_amount * sum(target_sz_old)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    # target_pos 表示的是前一帧的目标的位置
    x_crop_old = get_subwindow_tracking(im, target_pos_old, p.instance_size,
                                    round(s_x), avg_chans).unsqueeze(0)

    viz.image(x_crop_old.squeeze(), opts={"title":"search region"}, win="search region")
    # 这里的 target_pos 表示的是后一帧的目标新的位置
    target_pos_new, target_sz_new, score = tracker_eval(im, avg_chans, net, x_crop_old.cpu(), target_pos_old, target_sz_old * scale_z, window, scale_z, p)

    target_pos_new[0] = max(0, min(state['im_w'], target_pos_new[0]))
    target_pos_new[1] = max(0, min(state['im_h'], target_pos_new[1]))
    target_sz_new[0] = max(10, min(state['im_w'], target_sz_new[0]))
    target_sz_new[1] = max(10, min(state['im_h'], target_sz_new[1]))

    state['target_pos'] = target_pos_new
    state['target_sz'] = target_sz_new
    state['score'] = score
    return state

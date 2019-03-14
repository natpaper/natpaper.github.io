# Negative-Aware Training

<div style="display: flex;justify-content:center; align-items:Center;">
    <video muted src="natgan3.mp4" type="video/mp4" controls="" autoplay="autoplay" loop="loop" width="1000px" height="600px">
    </video>
</div>

**Please use Chrome to display video.**

## 1D GAN

Code of 1D GAN is based on [gan-intro](https://github.com/AYLIEN/gan-intro)

The two negative distributions are:

``` python
class NegDist(object):
    def __init__(self):
        self.mu = 13
        self.sigma = 1

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

class NegDist2(object):
    def __init__(self):
        self.mu = -6
        self.sigma = 2

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples
```

## NAT on supervised classification

Our supervised classification code on CIFAR 10 is based on [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

We implement the criterion:

``` python
class NATLoss(nn.Module):
    def __init__(self):
        super(NeutralCE, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, input, target, phase='test'):
        if phase != 'train':
            return self.ce(input, target)
        pos_idx = target != -1
        neg_idx = target == -1
        ce_loss = self.ce(input[pos_idx], target[pos_idx])
        neg_loss = self.kl(self.logsoftmax(input[neg_idx]), torch.tensor([[0.1] * 10]).repeat(input[neg_idx].shape[0], 1).to(device))
        return ce_loss * sum(pos_idx) / input.shape[0] + neg_loss * sum(neg_idx) / input.shape[0]
```

Note, if you use BCE loss in pytorch, the network performance on negative samples is slightly weaker compared with our results in paper.<br>
The ILSVRC2012 samples used in paper are:
```
n01514668, n01440764, n01484850, n01494475, n01496331, n01443537, n01491361, n01498041, n01675722, 
n01629819, n01630670, n01632458, n01631663, n01632777, n01689811, n01693334, n01675722, n01694178, 
n01692333, n01694178, n01695060, n01688243, n01693334, n01689811, n01728572, n01751748, n01755581, 
n01749939, n01753488, n01756291, n01773797, n01774384, n01775062, n01774750, n01776313, n01774384
```

## NAT-GAN

For NAT-GAN on CIFAR 10, code will be available soon, and you can easily implement it by modifying 
[AM GAN](https://github.com/ZhimingZhou/AM-GAN2), some important codes are:

``` python
fake_logits = discriminator(fake_datas, num_logits)
fake_logits2 = discriminator(tf.concat([fake_datas, next(neg_gen)[0]], 0), num_logits)

dis_fake_loss = kl_divergence(tf.ones_like(fake_logits, tf.float32) / num_logits, tf.nn.softmax(fake_logits))
dis_fake_loss2 = kl_divergence(tf.ones_like(fake_logits2, tf.float32) / num_logits, tf.nn.softmax(fake_logits2))
```

Set class number=10, batchsize=256, and you'd better add negative samples every 10 batches to allow model converge better.

There are some generated samples:

NAT-GAN: ![nat.png](nat.png)

NAT-GAN_1: ![nat_1.png](nat_1.png)

NAT-GAN_3: ![nat_3.png](nat_3.png)

NAT-GAN_10: ![nat_10.png](nat_10.png)

CatGAN: ![catgan.png](catgan.png)

CatGAN_1: ![catgan_1.png](catgan_1.png)

CatGAN_3: ![catgan_3.png](catgan_3.png)

CatGAN_10: ![catgan_10.png](catgan_10.png)

AM GAN: ![amgan.png](amgan.png)

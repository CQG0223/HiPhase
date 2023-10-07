from visdom import Visdom
import time
import numpy as np

import pdb
class Logger(object):
    def __init__(self):
        '''
        n_epochs : 
        batch_epoch : 
        '''
        self.viz = Visdom() #默认env是main
        self.batches_epoch = 1
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_preiod = 0
        self.losses = {}
        self.loss_windows = {} #保存loss图的字典组合
        self.image_windows = {} #保存生成图的字典集合
    
    def log(self,losses = None,images = None):
        self.mean_preiod += (time.time() - self.prev_time)
        self.prev_time = time.time()
        
        for i , loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name]
            else:
                self.losses[loss_name] = losses[loss_name]
            
            for image_name, tensor in images.items(): # 字典.items()是以list形式返回键值对
               
                if image_name not in self.image_windows:
                    img = self.__tensor2image(tensor.data)
                    self.image_windows[image_name] = self.viz.image(img, opts={'title':image_name})
                else:
                    self.viz.image(self.__tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

            # End of each epoch
            if (self.batch % self.batches_epoch) == 0: # 一个epoch结束时
                # 绘制loss曲线图
                for loss_name, loss in self.losses.items():
                    if isinstance(loss, np.ndarray):
                        if loss_name not in self.loss_windows:
                            
                            self.loss_windows[loss_name] = self.viz.line(X=np.arange(len(loss[0])) , Y=np.transpose(loss), opts={'xlabel':'epochs', 'ylabel':loss_name, 'title':loss_name})
                        else:
                            self.viz.line(X=np.arange(len(loss[0])) , Y=np.transpose(loss),win=self.loss_windows[loss_name], update='replace')
                    else:
                        if loss_name not in self.loss_windows:
                            self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                                                                        opts={'xlabel':'epochs', 'ylabel':loss_name, 'title':loss_name})
                        else:
                            self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append') #update='append'可以使loss图不断更新
                    # 每个epoch重置一次loss
                    self.losses[loss_name] = 0.0
                # 跑完一个epoch，更新一下下面参数
                self.epoch += 1
                self.batch = 1
            else:
                self.batch += 1

    def __tensor2image(self,tensor,imtype = np.float32):
        max = tensor.max()
        min = tensor.min()
        res = (tensor - min) / (max - min)
        image_numpy = res[0].cpu().float().numpy()
        return image_numpy.astype(imtype)
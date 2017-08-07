import time
import logging
import mxnet as mx


class Speedometer(object):
    def __init__(self, batch_size, frequent=10):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = True
        self.tic = 0
        self.last_count = 0
        #self.loss = model.get_params()[0]['loss8_313_output'].asnumpy()
        #print 'result: ', self.loss

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\nTrain-" % (param.epoch, count, speed)
                    for n, v in zip(name, value):
                        if 'Loss' in n or 'loss' in n:
                            s += "%s=%f\n" % (n, v)
                else:
                    s = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)
                s = s.strip()
                
                logging.info(s)
                print(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        weight = arg['rfcn_bbox_weight']
        bias = arg['rfcn_bbox_bias']
        repeat = bias.shape[0] / means.shape[0]

        arg['rfcn_bbox_weight_test'] = weight * mx.nd.repeat(mx.nd.array(stds), repeats=repeat).reshape((bias.shape[0], 1, 1, 1))
        arg['rfcn_bbox_bias_test'] = arg['rfcn_bbox_bias'] * mx.nd.repeat(mx.nd.array(stds), repeats=repeat) + mx.nd.repeat(mx.nd.array(means), repeats=repeat)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop('rfcn_bbox_weight_test')
        arg.pop('rfcn_bbox_bias_test')
    return _callback

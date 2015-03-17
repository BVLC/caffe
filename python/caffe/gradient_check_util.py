import numpy as np
import caffe


class GradientChecker:

    def __init__(self, stepsize, threshold, seed=1701, kink=0., kink_range=-1):
        for k, v in locals().iteritems():
            if k == 'self':
                continue
            self.__dict__[k + '_'] = v
        pass

    def get_obj_and_gradient(self, layer, top, top_id, top_data_id):
        for b in top:
            b.diff[...] = 0
        loss_weight = 2
        loss = top[top_id].data.flat[top_data_id] * loss_weight
        top[top_id].diff.flat[top_data_id] = loss_weight
        return loss

    def check_gradient_single(
            self, layer, bottom, top, check_bottom='all', top_id=0,
            top_data_id=0):
        """"""
        # Retrieve Blobs to check
        propagate_down = [False for i in xrange(len(bottom))]
        blobs_to_check = []
        for blob in layer.blobs:
            blobs_to_check += [blob]
        if check_bottom == 'all':
            check_bottom = range(len(bottom))
        assert len(check_bottom) <= len(bottom)
        for cb in check_bottom:
            blobs_to_check += [bottom[cb]]
            propagate_down[cb] = True

        # Compute the gradient analytically using Backward
        caffe.set_random_seed(self.seed_)
        layer.Reshape(bottom, top)
        layer.Forward(bottom, top)
        self.get_obj_and_gradient(layer, top, top_id, top_data_id)
        layer.Backward(top, propagate_down, bottom)

        # Store computed diff
        ana_grads = [b.diff.copy() for b in blobs_to_check]

        # Compute finite diff
        for bi, (ana_grad, blob) in enumerate(zip(ana_grads, blobs_to_check)):
            for fi in xrange(blob.count):
                step = self.stepsize_
                # L(fi <-- fi+step)
                blob.data.flat[fi] += step
                caffe.set_random_seed(self.seed_)
                layer.Reshape(bottom, top)
                layer.Forward(bottom, top)
                ploss = self.get_obj_and_gradient(
                    layer, top, top_id, top_data_id)
                # L(fi <-- fi-step)
                blob.data.flat[fi] -= 2 * step
                caffe.set_random_seed(self.seed_)
                layer.Reshape(bottom, top)
                layer.Forward(bottom, top)
                nloss = self.get_obj_and_gradient(
                    layer, top, top_id, top_data_id)
                grad = (ploss - nloss) / (2. * step)
                agrad = ana_grad.flat[fi]
                feat = blob.data.flat[fi]
                if self.kink_ - self.kink_range_ > np.abs(feat) \
                        or np.abs(feat) > self.kink_ + self.kink_range_:
                    scale = max(
                        max(np.abs(agrad), np.abs(grad)), 1.0)
                    assert np.isclose(
                        agrad, grad, rtol=0, atol=self.threshold_ * scale), (
                        "(top_id, top_data_id, blob_id, feat_id)"
                        "=(%d, %d, %d, %d); feat=%g; "
                        "objective+ = %g; objective- = %g; "
                        "analitical_grad=%g; finite_grad=%g" % (
                            top_id, top_data_id, bi, fi, feat, ploss, nloss,
                            agrad, grad)
                    )

    def check_gradient_exhaustive(
            self, layer, bottom, top, check_bottom='all'):
        """"""
        layer.SetUp(bottom, top)
        assert len(top) > 0
        for i in xrange(len(top)):
            for j in xrange(top[i].count):
                self.check_gradient_single(
                    layer, bottom, top, check_bottom, i, j)

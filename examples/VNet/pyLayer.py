import caffe
import numpy as np

class DiceLoss(caffe.Layer):
    """
    Compute energy based on dice coefficient.
    """
    union = None
    intersection = None
    result = None
    gt = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the dice. the result of the softmax and the ground truth.")



    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != 2*bottom[1].count:
            print bottom[0].data.shape
            print bottom[1].data.shape
            raise Exception("the dimension of inputs should match")

        # loss output is two scalars (mean and std)
        top[0].reshape(1)

    def forward(self, bottom, top):

        dice = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.union = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.intersection = np.zeros(bottom[0].data.shape[0],dtype=np.float32)

        self.result = np.reshape(np.squeeze(np.argmax(bottom[0].data[...],axis=1)),[bottom[0].data.shape[0],bottom[0].data.shape[2]])
        self.gt = np.reshape(np.squeeze(bottom[1].data[...]),[bottom[1].data.shape[0],bottom[1].data.shape[2]])

        self.gt = (self.gt > 0.5).astype(dtype=np.float32)
        self.result = self.result.astype(dtype=np.float32)

        for i in range(0,bottom[0].data.shape[0]):
            # compute dice
            CurrResult = (self.result[i,:]).astype(dtype=np.float32)
            CurrGT = (self.gt[i,:]).astype(dtype=np.float32)

            self.union[i]=(np.sum(CurrResult) + np.sum(CurrGT))
            self.intersection[i]=(np.sum(CurrResult * CurrGT))

            dice[i] = 2 * self.intersection[i] / (self.union[i]+0.00001)
            print dice[i]

        top[0].data[0]=np.sum(dice)

    def backward(self, top, propagate_down, bottom):
        for btm in [0]:
            prob = bottom[0].data[...]
            bottom[btm].diff[...] = np.zeros(bottom[btm].diff.shape, dtype=np.float32)
            for i in range(0, bottom[btm].diff.shape[0]):

                bottom[btm].diff[i, 0, :] += 2.0 * (
                    (self.gt[i, :] * self.union[i]) / ((self.union[i]) ** 2) - 2.0*prob[i,1,:]*(self.intersection[i]) / (
                    (self.union[i]) ** 2))
                bottom[btm].diff[i, 1, :] -= 2.0 * (
                    (self.gt[i, :] * self.union[i]) / ((self.union[i]) ** 2) - 2.0*prob[i,1,:]*(self.intersection[i]) / (
                    (self.union[i]) ** 2))

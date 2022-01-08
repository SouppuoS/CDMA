""" 
Implement of Circular Differential Microphone Arrays (CDMA)

Reference: 
[1] Benesty, Jacob, Jingdong Chen, and Israel Cohen. Design of circular differential microphone arrays. Vol. 12. Berlin, Germany:: Springer, 2015.
"""

import numpy as np
import matplotlib.pyplot as plt

c = 340

class circular_microphone_arrays():
    def __init__(self,
        r       : float     = 1.0,      # r         : radius of array (cm)
        M       : int       = 4,        # M         : number of microphones
        fs      : int       = 8000,     # fs        : sample rate
        f_bin   : int       = 256,      # f_bin     : number of freq bins
        sa_bin  : int       = 360,      # sa_bin    : number of steer angle
        ) -> None:
        self.r        = r / 100.        # cm -> m
        self.M        = M
        self.fs       = fs

        self.f_bin    = f_bin
        self.sa_bin   = sa_bin

        self.phi      = np.arange(0, self.M, 1) * 2 * np.pi / self.M        
        self.rad      = np.arange(0, 360, 360. / self.sa_bin) * np.pi / 180.
        self.freq     = np.arange(0, self.fs / 2, self.fs / 2 / self.f_bin)         # [f_bin]
        _cos_degdif   = np.cos(self.rad[:,None] - self.phi[None])                   # [sa_bin x M]
        self.steer    = np.exp(2j * np.pi * self.r / c \
                                  * _cos_degdif[...,None] * self.freq[None, None])  # [sa_bin x M x f_bin]

    @classmethod
    def theta_validation(
        self,
        theta: float,
    ) -> float :
        while theta < 0:
            theta += 360.
        while theta >= 360.:
            theta -= 360.
        return theta

    def get_steer(
        self,
        theta: float = 0    # theta: steer direction (degree)
    ) -> np.ndarray:
        theta_val = self.theta_validation(theta)
        print(f'LOG:: get steer from theta = {theta_val}')
        rad       = theta_val / 180. * np.pi
        tgt_theta = np.argmin(np.abs(self.rad - rad))
        return self.steer[tgt_theta]    # [M x f_bin]

class CDMA():
    """Circular Differential Microphone Arrays (CDMA)
    The order of CDMA depend on the number of null point in null_list.
    """
    def __init__(
        self,
        cma         : circular_microphone_arrays,
        sa          : float,                                # steer angle (distortless directgion)
        sym         : np.ndarray                 = None,    # symmetric constrain for the weight, [C, M + 1]
        null_list   : list                       = [],      # list of null point (degree)
        b           : np.ndarray                 = None,    # b for each point, [len(null_list) + 1]
        mic_mask    : list                       = [],      # list of microphone mask
        ) -> None:
        self.cma        = cma
        self.null_list  = null_list
        self.mic_mask   = mic_mask
        self.sa         = sa + 360 if sa < 0 else sa
        self.b          = b
        self.sym        = self.build_sym(self.cma.M, self.sa) if sym is None else sym

        self.calc_weight()
    
    def calc_weight(self):
        _eq     = np.array([self.cma.get_steer(d).conjugate() for d in [self.sa,] + self.null_list])
        _b      = np.zeros((_eq.shape[0], 1, _eq.shape[-1]))    # [N x 1 x f_bin]
        _b[0]   = 1.

        if self.b is not None:
            _b  = self.null_b[:, None, None].repeat(_eq.shape[-1], axis=-1)

        if self.sym is not None:
            _sy = self.sym[..., None].repeat(self.cma.f_bin, axis=-1).astype(np.float)
            _eq = np.concatenate((_eq, _sy[..., :-1, :]), axis=0)  # [N x M x f_bin], N : number of constrains
            _b  = np.concatenate((_b, _sy[..., [-1], :]), axis=0)  # [N x 1 x f_bin], N : number of constrains

        # DC part
        _eq[...,0]  = np.eye(self.cma.M)[:_eq.shape[0]]
        _b[...,0]   = 1. / self.cma.M

        # weight, [f_bin x M x 1]
        if _eq.shape[0] < _eq.shape[1]:
            # NMS
            _mAA        = np.einsum('ijk,ljk->kil', _eq, _eq.conjugate())  # [f_bin x N x N]
            self.weight = np.einsum('ijk,kil,lnk->kjn', _eq.conjugate(), np.linalg.inv(_mAA), _b)

        elif _eq.shape[0] == _eq.shape[1]:
            self.weight = np.linalg.solve(_eq.transpose(2, 0, 1), _b.transpose(2, 0, 1))

        else:
            raise NotImplemented

    @classmethod
    def build_sym(
        self,
        M           : int,              # M         : number of microphones
        sa          : float = 0,        # steer angle (distortless directgion)
    ) -> np.ndarray:
        sa       = sa + 360 if sa < 0 else sa
        deg_cand = np.array([v * 360 / M for v in range(M)])
        tgt_dir  = np.argmin(np.abs(deg_cand - sa))

        _sy = np.zeros((M // 2 - 1, M + 1), dtype=np.float)
        for i in range(1, M // 2):
            _sy[i - 1, i], _sy[i - 1, M - i] = 1., -1.
        _sy_r = np.concatenate((np.roll(_sy[:, :-1], shift=tgt_dir, axis=1), _sy[:, [-1]]), axis=1)
        return _sy_r

    def get_weight(self):
        return self.weight

if __name__ == "__main__":
    pass
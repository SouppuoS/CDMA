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

    def get_steer(
        self,
        theta: float = 0    # theta: steer direction (degree)
        ):
        rad       = theta / 180. * np.pi
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
        symmetric   : np.ndarray,                           # symmetric constrain for the weight
        sym_b       : np.ndarray                 = None,    # b part of symmetric constrain
        null_list   : list                       = [],      # list of null point (degree)
        mic_mask    : list                       = [],      # list of microphone mask
        min_norm    : bool                       = False,   # use minimum-norm filter
        ) -> None:
        self.cma        = cma
        self.null_list  = null_list
        self.mic_mask   = mic_mask
        
        _sy = symmetric[None, :, None] if len(symmetric.shape) == 1 else symmetric[..., None]
        _sy = _sy.repeat(cma.f_bin, axis=-1).astype(np.float)
        _eq = []
        for d in [sa,] + self.null_list:
            _eq.append(self.cma.get_steer(d))
        _eq         = np.array(_eq)
        _eq         = np.concatenate((_eq, _sy), axis=0)            # [N x M x f_bin], N : number of constrains
        _eq[...,0]  = np.eye(self.cma.M)[:_eq.shape[0]]
        _b          = np.zeros((_eq.shape[0], 1, _eq.shape[-1]))    # [N x 1 x f_bin]
        _b[0]       = 1.
        _b[...,0]   = 1. / self.cma.M

        if sym_b is not None:
            n_sym_b = sym_b.shape[-1]
            for i in range(n_sym_b):
                _b[- n_sym_b + i] = sym_b[i]
        
        if len(mic_mask) > 0:
            _eq = _eq[mic_mask]
            _eq = _eq[:, mic_mask]
            _b  = _b[mic_mask]

        if min_norm:
            _mAA        = np.einsum('ijk,ljk->kil', _eq, _eq.conjugate())  # [f_bin x N x N]
            self.weight = np.einsum('ijk,kil,lnk->kjn', _eq.conjugate(), np.linalg.inv(_mAA), _b)
        else:
            # [f_bin x M x 1]
            self.weight = np.linalg.solve(_eq.conjugate().transpose(2, 0, 1), _b.transpose(2, 0, 1))
    
    def get_weight(self):
        return self.weight

def Beampattern(
    cma     : circular_microphone_arrays,
    w       : np.ndarray,                           # [f_bin x M x 1]
    freq    : int = 0                               # target freq
):
    tgt_f_bin = np.argmin(np.abs(cma.freq - freq))
    # [sa_bin x M] x [M x 1]
    a_map     = np.einsum('ij,jk->ik', cma.steer[..., tgt_f_bin], w[tgt_f_bin].conjugate())
    plt.subplot(1,1,1, polar=True)
    plt.plot(np.linspace(0, 2 * np.pi, 360), 10 * np.log10(np.abs(a_map).reshape(-1) + 1e-2))
    plt.title('4-MIC CDMA 2nd-cardioid\nsteer angle = 90 deg null point at 180 deg and 270 deg')
    plt.tight_layout()
    plt.savefig('beampattern')

if __name__ == "__main__":
    cma     = circular_microphone_arrays(M=4)
    cdma    = CDMA(cma, sa=90, null_list=[225], 
                   symmetric=np.array([[-1, 0, 1, 0]]),
                   sym_b=np.array([[0]]), min_norm=True)
    Beampattern(cma, cdma.get_weight(), freq=1000)
    pass
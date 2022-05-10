""" 
Implement of Circular Differential Microphone Arrays (CDMA)

Reference: 
[1] Benesty, Jacob, Jingdong Chen, and Israel Cohen. Design of circular differential microphone arrays. Vol. 12. Berlin, Germany:: Springer, 2015.
[2] Benesty, Jacob, Jingdong Chen, and Yiteng Huang. Microphone array signal processing. Vol. 1. Springer Science & Business Media, 2008.
"""

import numpy as np

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

        # steer
        self.phi      = np.arange(0, self.M, 1) * 2 * np.pi / self.M        
        self.rad      = np.arange(0, 360, 360. / self.sa_bin) * np.pi / 180.
        self.freq     = np.linspace(0, self.fs / 2, self.f_bin)                     # [f_bin]
        self.freq[0]  = 1                                                           # set the first freq to 1Hz to avoid singular matrix
        _cos_degdif   = np.cos(self.rad[:,None] - self.phi[None])                   # [sa_bin x M]
        self.steer    = np.exp(2j * np.pi * self.r / c \
                                  * _cos_degdif[...,None] * self.freq[None, None])  # [sa_bin x M x f_bin]

        # gamma_dn
        _dij    = np.arange(0, self.M, 1)[None] - np.arange(0, self.M, 1)[:, None]  # delta_{ij}, [M x M]
        _delta  = 2 * self.r * np.sin(np.pi * _dij / self.M)
        self.g_dn_sphere   = np.sinc(2 * np.pi * self.freq[:, None, None] * _delta[None] / c)
        self.g_dn_cylinder = np.i0(2 * np.pi * self.freq[:, None, None] * _delta[None] / c)

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

    def get_steer(
        self,
        theta: float = 0    # theta: steer direction (degree)
    ) -> np.ndarray:
        theta_val = self.theta_validation(theta)
        rad       = theta_val / 180. * np.pi
        tgt_theta = np.argmin(np.abs(self.rad - rad))
        print(f'LOG:: get steer from theta = {theta_val} - tgt = {tgt_theta}')
        return self.steer[tgt_theta]    # [M x f_bin]

class FixedBeamformor():
    def __init__(
        self,
        ) -> None:
        self.flg_calc_weight = True

    def calc_weight(
        self
    ) -> None:
        pass

    def get_weight(
        self,
        recalc : bool = False
    ) -> np.ndarray:
        if self.flg_calc_weight or recalc:
            self.calc_weight()
            self.flg_calc_weight = False
        # weight, [f_bin x M x 1]
        return self.weight
    
    def apply(
        self, 
        spec_x  : np.ndarray,   # input signal, [M x f_bin x N]
    ) -> np.ndarray:
        return np.einsum('mfn,fmi->fni', spec_x, self.get_weight().conjugate())[..., 0]

class AdaptiveBeamformor():
    def __init__(
        self
    ) -> None:
        pass

    def get_Rxx(
        self,
        spec_x  : np.ndarray,
    ) -> np.ndarray:
        # Rxx, [f_bin x M x M]
        return np.einsum('mfn,lfn->fml', spec_x, spec_x.conjugate())
    
    def calc_weight(
        self,
        spec_x  : np.ndarray,
    ) -> np.ndarray:
        # weight, [f_bin x M x 1]
        pass

    def apply(
        self,
        spec_x  : np.ndarray,
    ) -> np.ndarray:
        print('LOG:: Adaptive')
        return np.einsum('mfn,fmi->fni', spec_x, self.calc_weight(spec_x).conjugate())[..., 0]

class CDMA(FixedBeamformor):
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
        super(CDMA, self).__init__()
        self.cma        = cma
        self.null_list  = null_list
        self.mic_mask   = mic_mask
        self.sa         = sa + 360 if sa < 0 else sa
        self.b          = b
        self.sym        = self.cma.build_sym(self.cma.M, self.sa) if sym is None else sym
    
    def build_eq(
        self,
    ) -> tuple:
        _eq     = np.array([self.cma.get_steer(d).conjugate() for d in [self.sa,] + self.null_list])
        _b      = np.zeros((_eq.shape[0], 1, _eq.shape[-1]))    # [N x 1 x f_bin]
        _b[0]   = 1.

        if self.b is not None:
            _b  = self.null_b[:, None, None].repeat(_eq.shape[-1], axis=-1)

        _sy = self.sym[..., None].repeat(self.cma.f_bin, axis=-1).astype(np.float)
        _eq = np.concatenate((_eq, _sy[..., :-1, :]), axis=0)  # [N x M x f_bin], N : number of constrains
        _b  = np.concatenate((_b, _sy[..., [-1], :]), axis=0)  # [N x 1 x f_bin], N : number of constrains

        # DC part
        _eq[...,0]  = np.eye(self.cma.M)[:_eq.shape[0]]
        _b[...,0]   = 1. / self.cma.M
        return _eq, _b

    def calc_weight(
        self,
    ) -> None:
        _eq, _b = self.build_eq()
        if _eq.shape[0] < _eq.shape[1]:
            # NMS
            print('LOG:: NMS mode')
            _mAA        = np.einsum('ijk,ljk->kil', _eq, _eq.conjugate())  # [f_bin x N x N]
            self.weight = np.einsum('ijk,kil,lnk->kjn', _eq.conjugate(), np.linalg.inv(_mAA), _b)

        elif _eq.shape[0] == _eq.shape[1]:
            self.weight = np.linalg.solve(_eq.transpose(2, 0, 1), _b.transpose(2, 0, 1))

        else:
            raise NotImplemented

class DS(FixedBeamformor):
    """Delay and Sum
    """
    def __init__(
        self,
        cma         : circular_microphone_arrays,
        sa          : float,                                # steer angle (distortless directgion)
        ) -> None:
        self.cma        = cma
        self.sa         = sa + 360 if sa < 0 else sa

    def calc_weight(
        self,
    ) -> None:
        self.weight = self.cma.get_steer(self.sa).T[...,None] / self.cma.M

class RSD(CDMA):
    """Robust Superdirective beamforming
    """
    def __init__(
        self,
        cma         : circular_microphone_arrays,
        sa          : float,                                # steer angle (distortless directgion)
        mode        : str                   = 'sphere',     # diffuse noise mode, `sphere` or `cylinder`
        sym_flag    : bool                  = False,        # add symmetry constraint or not
        eps = 0.,                                           # eps on Gamma_dn, could be `float` or `np.ndarray` for [f_bin x 1 x 1]
        **kwargs,
        ) -> None:
        super(RSD, self).__init__(cma, sa, **kwargs)
        self.g_dn_mode  = mode
        self.eps        = eps
        self.flg_sym    = sym_flag

    def calc_weight(
        self,
    ) -> None:
        _g_v = self.eps * np.eye(self.cma.M)[None] + (self.cma.g_dn_sphere \
               if self.g_dn_mode == 'sphere' else self.cma.g_dn_cylinder)                   # [f_bin x M x M]
        if not self.flg_sym:
            # without symmetry constraint
            sv          = self.cma.get_steer(self.sa)                                       # [M x f_bin]
            _num        = np.einsum('fmn,imf->fni', np.linalg.inv(_g_v), sv[None])          # [f_bin x M x 1]
            _den        = np.einsum('fmi,lmf->fil', _num, sv[None].conjugate())             # [f_bin x 1 x 1]
            _num[0]     = 1 / self.cma.M                                                    # set DC part to zero
            self.weight = _num / _den
        
        else:
            # without symmetry constraint
            _eq, _b     = self.build_eq()
            _gd_inv     = np.linalg.solve(_g_v, _eq.conjugate().transpose(2, 1, 0))     # \Gamma^{-1}D^*, [f_bin x M x N]
            _d_gd_inv   = np.einsum('nmf,fml->fnl', _eq, _gd_inv)                       # D\Gamma^{-1}D^*, [f_bin x N x N]
            self.weight = np.einsum('fmn,fnl,lif->fmi', _gd_inv, np.linalg.inv(_d_gd_inv), _b)

class GSC(AdaptiveBeamformor, CDMA):
    """Generalized Sidelobe Canceler
    """
    def __init__(
        self,
        cma     : circular_microphone_arrays,
        inv_eps : float                         = 1e-6,
        *args,
        **kwargs) -> None:
        AdaptiveBeamformor.__init__(self)
        CDMA.__init__(self, cma, *args, **kwargs)
        
        self.inv_eps = inv_eps

    def calc_weight(
        self,
        spec_x  : np.ndarray,
    ) -> None:
        _eq, _b     = self.build_eq()
        _Rxx        = self.get_Rxx(spec_x)
        N, M, _     = _eq.shape
        _mAA        = np.einsum('ijk,ljk->kil', _eq, _eq.conjugate())  # [f_bin x N x N]
        _w_dma      = np.einsum('ijk,kil,lnk->kjn', _eq.conjugate(), np.linalg.inv(_mAA), _b)
        _proj       = np.einsum('ijk,kil,lnk->kjn', _eq.conjugate(), np.linalg.inv(_mAA), _eq)
        _bc         = (np.eye(M) - _proj)[..., :N]
        _bcRxxbcT   = np.einsum('fmn,fmm,fml->fnl', _bc.conjugate(), _Rxx, _bc)
        _w_gsc      = np.linalg.solve(_bcRxxbcT + self.inv_eps * np.eye(N)[None], 
                                      np.einsum('fmn,fmm,fml->fnl', _bc.conjugate(), _Rxx, _w_dma))
        return _w_dma - np.einsum('fmn,fnl->fml', _bc, _w_gsc)

if __name__ == "__main__":
    pass
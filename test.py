# test code for CDMA
import numpy as np
import matplotlib.pyplot as plt
from cdma import circular_microphone_arrays, CDMA

def Beampattern(
    cma     : circular_microphone_arrays,
    w       : np.ndarray,                           # [f_bin x M x 1]
    freq    : int = 0                               # target freq
):
    tgt_f_bin = np.argmin(np.abs(cma.freq - freq))
    a_map     = np.einsum('ij,jk->ik', cma.steer[..., tgt_f_bin], w[tgt_f_bin].conjugate())
    plt.figure(figsize=(5,5))
    plt.subplot(1,1,1, polar=True)
    plt.plot(np.linspace(0, 2 * np.pi, 360), 10 * np.log10(np.clip(np.abs(a_map).reshape(-1), a_min=1e-2, a_max=None)))
    plt.ylim((-20, 0))
    plt.yticks([0, -5, -10, -15, -20], [0, -5, -10, -15, -20])
    plt.tight_layout()

def test_2ndOrder():
    sa      = 180
    cma     = circular_microphone_arrays(M=4)
    cdma    = CDMA(cma, sa=sa, null_list=[sa + 72, sa + 144])
    Beampattern(cma, cdma.get_weight(), freq=1000)
    plt.savefig('beampattern')

if __name__ == '__main__':
    test_2ndOrder()
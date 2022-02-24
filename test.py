# test code for CDMA
import numpy as np
import matplotlib.pyplot as plt

import pyroomacoustics as pra
import scipy.signal as ss
import soundfile as sf

from cdma import circular_microphone_arrays, CDMA, DS, RSD

def Beampattern(
    cma     : circular_microphone_arrays,
    w       : np.ndarray,                           # [f_bin x M x 1]
    freq    : int = 0                               # target freq
):
    tgt_f_bin = np.argmin(np.abs(cma.freq - freq))
    a_map     = np.einsum('ij,jk->ik', cma.steer[..., tgt_f_bin], w[tgt_f_bin].conjugate())
    return a_map

def Beampattern_allfreq(
    cma     : circular_microphone_arrays,
    w       : np.ndarray,                           # [f_bin x M x 1]
):
    a_map = np.einsum('dmf,fmi->dfi', cma.steer, w.conjugate())
    return a_map[..., 0]

def test_beampattern(cma, bf):
    a_map = Beampattern(cma, bf.get_weight(), freq=2000)
    plt.figure(figsize=(5,5))
    plt.subplot(1,1,1, polar=True)
    plt.plot(np.linspace(0, 2 * np.pi, cma.sa_bin), 20 * np.log10(np.clip(np.abs(a_map).reshape(-1), a_min=1e-2, a_max=None)))
    plt.ylim((-20, 0))
    plt.yticks([0, -5, -10, -15, -20], [0, -5, -10, -15, -20])
    plt.tight_layout()
    plt.savefig('beampattern')

    a_map_af = Beampattern_allfreq(cma, bf.get_weight()).transpose(1, 0)
    plt.figure()
    plt.imshow(20 * np.log10(np.clip(np.abs(a_map_af[1:]), a_min=1e-2, a_max=None)), aspect='auto')
    plt.colorbar()
    plt.ylabel('f_bin')
    plt.xlabel('Degree')
    plt.tight_layout()
    plt.savefig('beampattern_allfreq')

def test_enhance(
    cdma    : CDMA,
):
    s1_clean, fs = sf.read('/home/zhy/Data/dataset/thchs30/data_thchs30/train/C8_749.wav')
    s2_clean, _  = sf.read('/home/zhy/Data/dataset/thchs30/data_thchs30/train/C7_545.wav')
    T         = min(s1_clean.size, s2_clean.size)

    rt60      = 0.2
    room_sz   = [10., 10., 3.]
    e, odr    = pra.inverse_sabine(rt60, room_dim=room_sz)
    room      = pra.ShoeBox(room_sz, fs, materials=pra.Material(e), max_order=odr)

    r_ss      = 0.5
    dir_ss_1  = 180
    dir_ss_2  = 180 + 142
    r_micarray= cdma.cma.r
    room.add_source([room_sz[0] / 2 + r_ss * np.cos(dir_ss_1 / 180. * np.pi), 
                     room_sz[1] / 2 + r_ss * np.sin(dir_ss_1 / 180. * np.pi), 
                     room_sz[2] / 2], signal=s1_clean, delay=0.0)
    room.add_source([room_sz[0] / 2 + r_ss * np.cos(dir_ss_2 / 180. * np.pi),
                     room_sz[1] / 2 + r_ss * np.sin(dir_ss_2 / 180. * np.pi),
                     room_sz[2] / 2], signal=s2_clean, delay=0.0)
    room.add_microphone_array(
        np.c_[
            [room_sz[0] / 2 + r_micarray, room_sz[1] / 2, room_sz[2] / 2],
            [room_sz[0] / 2, room_sz[1] / 2 + r_micarray, room_sz[2] / 2],
            [room_sz[0] / 2 - r_micarray, room_sz[1] / 2, room_sz[2] / 2],
            [room_sz[0] / 2, room_sz[1] / 2 - r_micarray, room_sz[2] / 2],
        ]
    )

    s_rir              = room.simulate(return_premix=True)[..., :T]    # premix data, [S x M x T]
    mix_rir            = room.mic_array.signals[..., :T]               # mixed data,  [M x T]

    # enhance signal
    _, _, spec_mix_rir = ss.stft(mix_rir, fs, nperseg=(cdma.cma.f_bin - 1) * 2)
    _, s_enh           = ss.istft(cdma.apply(spec_mix_rir), fs)

    plt.figure()
    plt.subplot(2,2,1)
    plt.specgram(s_rir[0, 0, ...], Fs=fs)
    plt.title('Signal-1')
    plt.subplot(2,2,2)
    plt.specgram(s_rir[1, 0,...], Fs=fs)
    plt.title('Signal-2')
    plt.subplot(2,2,3)
    plt.specgram(mix_rir[0, ...], Fs=fs)
    plt.title('Mixture')
    plt.subplot(2,2,4)
    plt.specgram(s_enh, Fs=fs)
    plt.title('Enhanced signal-1')
    plt.tight_layout()
    plt.savefig('enhanced')

if __name__ == '__main__':
    sa      = 180
    cma     = circular_microphone_arrays(M=4, f_bin=129, r=1, fs=16000)

    cdma    = CDMA(cma, sa=sa, null_list=[sa + 72, sa + 144])
    test_beampattern(cma, cdma)
    test_enhance(cdma)

    # ds      = DS(cma, sa=sa)
    # test_beampattern(cma, ds)

    # sd      = RSD(cma, sa=sa, mode='sphere', eps=.0)
    # test_beampattern(cma, sd)
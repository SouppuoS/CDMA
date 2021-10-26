# CDMA
Generate weights for Differential Microphone Arrays (CDMA)[1] in python. Allow to design cardioid, supercardioid or hypercardioid beamformer.  
![cardioid2nd](beampattern.png)

## HOW TO USE
```
cma     = circular_microphone_arrays(M=4)
cdma    = CDMA(cma, sa=90, null_list=[180, 270], symmetric=np.array([[-1,0,1, 0]]))
Beampattern(cma, cdma.get_weight(), freq=1000)
```
- **2-STEP TO USE:** First, define a `circular_microphone_arrays` base on geometry of your microphone array, eg: `r` for radius (cm), `M` for the number of microphones. For now, only uniform circular array (UCA) is supported.  
Second, design your CDMA base on your microphone array, including `sa` for steer angle, `null_list` for null point list and `symmetric` for H property.  
The order of CDMA is depend on `null_list` and `symmetric`.  
For `symmetric` design, you can see *Section 2.1* and *Section 2.2* in [1] for help.  
  
- To build a **Minimum-norm Filter**, which is robust against white noise amplification, use the code below. The formulation of minimum-norm can be found in *Section 7.1* from [1].
```
cma     = circular_microphone_arrays(M=4)
cdma    = CDMA(cma, sa=90, null_list=[225], 
               symmetric=np.array([[-1, 0, 1, 0]]),
               sym_b=np.array([[0]]), min_norm=True)
```

## TO-DO
- mask for microphones  

## Reference:  
[1] Benesty, Jacob, Jingdong Chen, and Israel Cohen. Design of circular differential microphone arrays. Vol. 12. Berlin, Germany:: Springer, 2015.
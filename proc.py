import librosa
import os
import soundfile as sf
import numpy

target =  ['ht1', 'muff']
data_dir = 'Data'
target_dir = 'Data_proc'
sub_dir = ['test', 'train', 'val']
ext = ['input', 'target']

for s in sub_dir:
    for t in target:
        for e in ext:
            s_dir = os.path.join(data_dir, s, t)
            t_dir = os.path.join(target_dir, s, t)
            if(not os.path.exists(t_dir)):
                os.makedirs(t_dir)
            audio, sr = librosa.load(s_dir+'-'+e+'.wav')
            seg_len = (int)(sr * 0.5)
            for i in range(int(len(audio)/seg_len)):
                sf.write(os.path.join(t_dir, e)+str(i)+'.wav', numpy.array(audio[i*seg_len:(i+1)*seg_len]),sr)
            

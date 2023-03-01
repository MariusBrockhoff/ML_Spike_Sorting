"""
Inputs:
  file_name = name of file, string
  path = path of file location, string
  is_big_file = Is the file too big to be open at once (RAM limitations), boolean, default = False
    
Outputs:
  recording_data = np.array containing the raw recording data. Shape = (recorded data points, number of electrodes)
  electrode_stream = original data file containing all raw data and channel infos
  fsample = sampling frequency in [Hz]
    
"""
#If you want to use / import Multi channel Systems data, easiest way is to 
#import their custom functions to open the standard generated .h5 files:
# MCS PyData tools


#pip install McsPyDataTools
import McsPy
import McsPy.McsData
from McsPy import ureg, Q_


def file_opener(file_name, path, is_big_file=False):
    

    file_path = path + file_name + '.h5'
    
    big_file = is_big_file
    
    file = McsPy.McsData.RawData(file_path)
    electrode_stream = file.recordings[0].analog_streams[0]

    
    # extract basic information
    electrode_ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
    fsample = int(electrode_stream.channel_infos[0].sampling_frequency.magnitude)
    
    #Get signal
    if big_file:
      step_size = 10
      min_step = 0
      max_step = 60
      for i in range(min_step, max_step, step_size):
        #signal = get_channel_data(electrode_stream, channel_ids = [j for j in range(i,i+step_size)])
        scale_factor_for_uV = Q_(1,'volt').to(ureg.uV).magnitude
        if i == min_step:
          recording_data = (get_channel_data(electrode_stream, channel_ids = [j for j in range(i,i+step_size)]) * scale_factor_for_uV).T
        else:
          recording_data = np.concatenate((recording_data, (get_channel_data(electrode_stream, channel_ids = [j for j in range(i,i+step_size)]) * scale_factor_for_uV).T), axis=1)
        print("iteration", i+step_size, "completed")
        
    
    else:
      signal = get_channel_data(electrode_stream, channel_ids = [])
      scale_factor_for_uV = Q_(1,'volt').to(ureg.uV).magnitude
      recording_data = (get_channel_data(electrode_stream, channel_ids = []) * scale_factor_for_uV).T
    
    print("recording.shape:", recording_data.shape)
    
    return recording_data, electrode_stream, fsample

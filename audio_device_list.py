import pyaudio

audio = pyaudio.PyAudio()
print("list of available audio devices")
for i in range(0, audio.get_device_count()):
    data = audio.get_device_info_by_index(i)
    name = data['name']
    max_input_channels = data['maxInputChannels']
    max_output_channels = data['maxOutputChannels']
    asinput = 'Yes' if max_input_channels >= 1 else 'No'
    asoutput = 'Yes' if max_output_channels >= 1 else 'No'
    print(f"ID: {i}, Name: {name} [Input: {asinput} Output: {asoutput}]")

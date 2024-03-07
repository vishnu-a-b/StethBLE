from flask import Flask, render_template, request
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin, lfilter
import wave
from datetime import datetime

app = Flask(__name__)

# Set the upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_bandpass_filter(input_file_path, output_file_path, filter_order, lowcutf, highcutf):
    # Read the wave file
    sample_rate, data = wavfile.read(input_file_path)

    # Check if the data is mono or stereo
    num_channels = 1 if len(data.shape) == 1 else data.shape[1]

    # Calculate the time axis in seconds
    time = np.arange(0, len(data)) / sample_rate

    # Define bandpass filter parameters
    nyquist = 0.5 * sample_rate
    low = lowcutf / nyquist
    high = highcutf / nyquist
    b = firwin(filter_order, [low, high], pass_zero=False)

    # Apply the filter to each channel
    filtered_data = lfilter(b, 1, data, axis=0)

    # Plot the original and filtered waveforms
    plt.figure(figsize=(10, 6))

    # Plot the original waveform
    plt.subplot(2, 1, 1)
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original Waveform')

    # Plot the filtered waveform
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Filtered Waveform')

    plt.tight_layout()
    plt.savefig('static/plot.png')  # Save the plot as a static file

    # Convert the filtered data to integer format (16-bit PCM)
    filtered_data_int = (filtered_data * 32767).astype(np.int16)

    # Create a wave file and write the filtered data
    with wave.open(output_file_path, 'w') as wf:
        wf.setnchannels(1)  # Set to 1 for mono, 2 for stereo
        wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
        wf.setframerate(sample_rate)
        wf.writeframes(filtered_data_int.tobytes())

@app.route('/')
def index():
    return render_template('index.html', data={})

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        # Create the 'uploads' directory if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Save the uploaded file with a timestamp in the filename
        # timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input.wav")
        file.save(uploaded_file_path)

        # Get filter_order, lowcutf, and highcutf from the form
        filter_order = int(request.form.get('filter_order', 600))
        lowcutf = float(request.form.get('lowcutf', 0.5))
        highcutf = float(request.form.get('highcutf', 200))

        # Apply the bandpass filter and save the filtered output
        filtered_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output.wav")
        apply_bandpass_filter(uploaded_file_path, filtered_output_path, filter_order, lowcutf, highcutf)

        # Pass filter_order, lowcutf, and highcutf to the template
        data = {'filter_order': filter_order, 'lowcutf': lowcutf, 'highcutf': highcutf}

        return render_template('index.html', data=data)
    else:
        return 'Invalid file type'

if __name__ == '__main__':
    app.run(debug=True)

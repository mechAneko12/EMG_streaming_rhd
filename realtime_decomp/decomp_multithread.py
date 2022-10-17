import threading
import time
import sys
import socket

def readUint32(array, arrayIndex):
    variableBytes = array[arrayIndex : arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex = arrayIndex + 4
    return variable, arrayIndex

def readInt32(array, arrayIndex):
    variableBytes = array[arrayIndex : arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=True)
    arrayIndex = arrayIndex + 4
    return variable, arrayIndex

def readUint16(array, arrayIndex):
    variableBytes = array[arrayIndex : arrayIndex + 2]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex = arrayIndex + 2
    return variable, arrayIndex

class RealtimeEmgProcessor():
    def __init__(self, channel_names: list, numBlocks: int) -> None:
        self.event = threading.Event()
        self.scommand, self.swaveform, self.timestep = self._init_rhd(channel_names, numBlocks)
    
    def _init_rhd(channel_names: list, numBlocks: int):
        # Declare buffer size for reading from TCP command socket
        # This is the maximum number of bytes expected for 1 read. 1024 is plenty for a single text command
        COMMAND_BUFFER_SIZE = 1024 # Increase if many return commands are expected

        # Connect to TCP command server - default home IP address at port 5000
        print('Connecting to TCP command server...')
        scommand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        scommand.connect(('127.0.0.1', 5000))

        # Connect to TCP waveform server - default home IP address at port 5001
        print('Connecting to TCP waveform server...')
        swaveform = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        swaveform.connect(('127.0.0.1', 5001))

        # Query runmode from RHX software
        scommand.sendall(b'get runmode')
        commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
        isStopped = commandReturn == "Return: RunMode Stop"

        # If controller is running, stop it
        if not isStopped:
            scommand.sendall(b'set runmode stop')
            time.sleep(0.1) # Allow time for RHX software to accept this command before the next one comes

        # Query sample rate from RHX software
        scommand.sendall(b'get sampleratehertz')
        commandReturn = str(scommand.recv(COMMAND_BUFFER_SIZE), "utf-8")
        expectedReturnString = "Return: SampleRateHertz "
        if commandReturn.find(expectedReturnString) == -1: # Look for "Return: SampleRateHertz N" where N is the sample rate
            raise Exception('Unable to get sample rate from server')
        else:
            sampleRate = float(commandReturn[len(expectedReturnString):])

        # Calculate timestep from sample rate
        timestep = 1 / sampleRate

        # Clear TCP data output to ensure no TCP channels are enabled
        scommand.sendall(b'execute clearalldataoutputs')
        time.sleep(0.1)

        # Send TCP commands to set up TCP Data Output Enabled for wide
        for channel_name in channel_names:
            for i in range(64):
                channel_index = str(i).rjust(3, '0')
                _command = f'set {channel_name}-{channel_index}.tcpdataoutputenabled true'
                scommand.sendall(_command.encode('utf-8'))
                time.sleep(0.1)

        # Calculations for accurate parsing
        # At 30 kHz with 1 channel, 1 second of wideband waveform data (including magic number, timestamps, and amplifier data) is 181,420 bytes
        # N = (framesPerBlock * waveformBytesPerFrame + SizeOfMagicNumber) * NumBlocks where:
        # framesPerBlock = 128 ; standard data block size used by Intan
        # waveformBytesPerFrame = SizeOfTimestamp + SizeOfSample ; timestamp is a 4-byte (32-bit) int, and amplifier sample is a 2-byte (16-bit) unsigned int
        # SizeOfMagicNumber = 4; Magic number is a 4-byte (32-bit) unsigned int
        # NumBlocks = NumFrames / framesPerBlock ; At 30 kHz, 1 second of data has 30000 frames. NumBlocks must be an integer value, so round up to 235

        numAmpChannels = len(channel_names) * 64
        framesPerBlock = 128
        waveformBytesPerFrame = 4 + 2 * numAmpChannels
        waveformBytesPerBlock = framesPerBlock * waveformBytesPerFrame + 4
        waveformBytesPerBlocks = numBlocks * waveformBytesPerBlock

        return scommand, swaveform, timestep
    
    def emg_getter(self):
        count = 0
        self.event.set()  # 初期値は青信号
        while True:
            if 5 < count <= 10:
                self.event.clear()  # 赤信号にする
                print("\33[41;1m赤信号...\033[0m")
            elif count > 10:
                self.event.set()  # 青信号にする
                count = 0
            else:
                print("\33[42;1m青信号...\033[0m")

            time.sleep(1)
            count += 1
    
    def emg_processor(self, name):
        while True:
            if self.event.is_set():  # 青信号がどうかをチェック
                print("[{}] 前進する...".format(name))
                time.sleep(1)
            else:
                print("[{}] 赤信号のため、信号を待つ...".format(name))
                self.event.wait()
                # flag=Trueになるまでここでブロッキングする
                print("[{}] 青信号のため、前進開始...".format(name))
    
    def main_thread(self):
        while True:
            try:
                _ = input()
            except KeyboardInterrupt:
                self.scommand.sendall(b'set runmode stop')
                time.sleep(0.1)
                print('end processor...')
                sys.exit()

if __name__ == '__main__':
    realtime_emg_processor = RealtimeEmgProcessor(['b', 'c'], 3)
    
    emg_getter = threading.Thread(target=realtime_emg_processor.emg_getter,)
    emg_getter.setDaemon(True)
    emg_getter.start()

    emg_processor = threading.Thread(target=realtime_emg_processor.emg_processor, args=("MINI",))
    emg_processor.setDaemon(True)
    emg_processor.start()

    realtime_emg_processor.main_thread()

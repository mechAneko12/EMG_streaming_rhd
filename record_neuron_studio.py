import time
import socket
import pandas as pd

# Axis Neuronを起動させてからつかうこと！


if __name__ == '__main__':    
    HOST = '127.0.0.1'
    PORT = 7001      #Axis Neuron
    

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST,PORT))
    print('connected')
    try:
        motion_tmp = []
        all_data = []
        s = time.time()
        print('start recording')
        while True:
            rx_meesage = client.recv(4096)
            motion = rx_meesage.decode(encoding='utf-8')
            motion = motion.split('||')[0]
            motion = [float(x.strip()) for x in motion[9:len(motion)].split()]
            if motion is not motion_tmp:
                timestamp = time.time() -s
                all_data.append([timestamp] + motion)
                motion_tmp = motion.copy()
                print(timestamp)
            
    except KeyboardInterrupt:
        columns = ['timestamp'] + [str(x) for x in range(354)]
        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv('sample.csv')
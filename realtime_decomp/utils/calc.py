import numpy as np

def rms(x: np.array) -> np.array: 
    """calc rms.

    Args:
        x (np.array): (n_sample, n_channel)

    Returns:
        np.array: (n_channel, )
    """
    return np.apply_along_axis(_calc_rms, 0, x)

def _calc_rms(x: np.array):
    return np.sqrt(np.square(x).mean())

if __name__ == '__main__':
    import time
    
    a = np.random.rand(768, 128)
    
    s = time.time()
    rms_a = rms(a)
    time_to_calc = time.time() - s
    print(f'input shape: {a.shape}\noutput shape: {rms_a.shape}\ncalc time: {time_to_calc}')
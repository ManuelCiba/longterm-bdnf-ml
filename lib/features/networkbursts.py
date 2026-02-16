import numpy as np
from lib.features.bursts import detect_bursts_isi
from lib.features.helper import robust_stats
import quantities as pq

def compute_networkburst_features(spiketrain_list, rec_dur_s=None):

    if len(spiketrain_list) == 0:
        return {}
    
    if rec_dur_s is None:
        rec_dur_s = float((spiketrain_list[0].t_stop - spiketrain_list[0].t_start).rescale(pq.s))
    

    bursts_per_channel = []
    for spiketrain in spiketrain_list:
        bursts = detect_bursts_isi(spiketrain)
        bursts_per_channel.append(bursts)

    network_bursts = detect_network_bursts(bursts_per_channel)
    features = network_burst_features(network_bursts, spiketrain_list, rec_dur_s)
    return features

def detect_network_bursts(bursts_per_channel, min_active_bursts=3, min_duration=0.1):
    """
    Detect network bursts using event-overlap method.
    
    Parameters
    ----------
    bursts_per_channel : dict
        {channel_index: list of bursts}, each burst = (start, end)
    min_active_bursts : int
        Minimum number of simultaneously bursting channels to define a network burst
    min_duration : float
        Minimum duration (s) to consider a valid network burst
    
    Returns
    -------
    network_bursts : list of tuples
        [(start, end), ...]
    """
    
    events = []
    for bursts in bursts_per_channel:
        for burst in bursts:
            start, end = burst["start"], burst["end"]
            events.append((start, +1))
            events.append((end, -1))
    events.sort(key=lambda x: (x[0], -x[1]))

    active = 0
    network_start = None
    network_bursts = []
    
    for t, delta in events:
        active += delta
        if active >= min_active_bursts and network_start is None:
            network_start = t
        elif active < min_active_bursts and network_start is not None:
            network_end = t
            if (network_end - network_start) >= min_duration:
                network_bursts.append((network_start, network_end))
            network_start = None
    
    # close last burst
    if network_start is not None:
        end_time = events[-1][0]
        if (end_time - network_start) >= min_duration:
            network_bursts.append((network_start, end_time))
    
    return network_bursts


def network_burst_features(network_bursts, spiketrains, rec_dur):
    """
    Compute standard network burst statistics from detected network bursts.
    
    Parameters
    ----------
    network_bursts : list of tuples
        [(start, end), ...]
    spiketrains : list of Neo SpikeTrain objects
    rec_dur : float
        Recording duration in seconds
    
    Returns
    -------
    features : dict
        {
            "nb_rate": float,
            "nb_d": float,
            "nb_ibi": float,
            "nb_sib": float,
            "nb_ae": float
        }
    """
    
    nb_durations = []
    nb_ibi = []
    nb_spikes = []
    nb_active_electrodes = []
    
    for i, (start, end) in enumerate(network_bursts):
        nb_durations.append(end - start)
        
        if i > 0:
            nb_ibi.append(start - network_bursts[i-1][1])
        
        spikes_in_burst = 0
        active_channels = 0
        for st in spiketrains:
            spikes = st.magnitude.flatten()
            n_spikes = np.sum((spikes >= start) & (spikes <= end))
            spikes_in_burst += n_spikes
            if n_spikes > 0:
                active_channels += 1
        
        nb_spikes.append(spikes_in_burst)
        nb_active_electrodes.append(active_channels)
    
    n_bursts = len(network_bursts)

    features = {}
    features["nb_rate"] = n_bursts / rec_dur if rec_dur > 0 else np.nan

    features["nb_d_med"], features["nb_d_iqr"], features["nb_d_cvrobust"] = robust_stats(nb_durations)
    features["nb_ibi_med"], features["nb_ibi_iqr"], features["nb_ibi_cvrobust"] = robust_stats(nb_ibi)
    features["nb_sib_med"], features["nb_sib_iqr"], features["nb_sib_cvrobust"] = robust_stats(nb_spikes)
    features["nb_ae_med"], features["nb_ae_iqr"], features["nb_ae_cvrobust"] = robust_stats(np.array(nb_active_electrodes)/len(spiketrains))
        
    return features
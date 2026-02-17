import numpy as np
from lib.features.bursts import detect_bursts_for_all_channel
from lib.features.helper import robust_stats
import quantities as pq
import matplotlib.pyplot as plt

def compute_networkburst_features(spiketrain_list, isi_threshold=0.1*pq.s, min_spikes_in_burst=3):

    if len(spiketrain_list) == 0:
        return {}
    
    rec_dur_s = float((spiketrain_list[0].t_stop - spiketrain_list[0].t_start).rescale(pq.s))
    
    bursts_per_channel = detect_bursts_for_all_channel(spiketrain_list, isi_threshold=isi_threshold, min_spikes_in_burst=min_spikes_in_burst)

    network_bursts = detect_network_bursts(bursts_per_channel)

    fig = plot_spiketrains_with_network_bursts(spiketrain_list, network_bursts)

    features = network_burst_features(network_bursts, spiketrain_list, rec_dur_s)
    
    return features, fig


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
    for ch, bursts in bursts_per_channel.items():
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
    features["rate"] = n_bursts / rec_dur if rec_dur > 0 else np.nan

    features["dur_med"], features["dur_iqr"], features["dur_rcv"] = robust_stats(nb_durations)
    features["ibi_med"], features["ibi_iqr"], features["ibi_rcv"] = robust_stats(nb_ibi)
    features["sib_med"], features["sib_iqr"], features["sib_rcv"] = robust_stats(nb_spikes)
    features["ae_med"], features["ae_iqr"], features["ae_rcv"] = robust_stats(np.array(nb_active_electrodes)/len(spiketrains))
        
    return features


def plot_spiketrains_with_network_bursts(
    spiketrain_list,
    network_bursts,
    figsize=(12, 6),
    spike_marker="|",
    spike_size=6,
    burst_color="red",
    burst_alpha=0.25,
    title="Spike trains with detected network bursts"
):
    """
    Plot raster of spike trains with network bursts overlay.

    Parameters
    ----------
    spiketrain_list : list of neo.SpikeTrain
        One SpikeTrain per electrode.
    network_bursts : list of tuples
        [(start, end), ...] network burst times in seconds.
    figsize : tuple
        Figure size.
    spike_marker : str
        Marker style for spikes.
    spike_size : int
        Marker size.
    burst_color : str
        Color of network burst shading.
    burst_alpha : float
        Transparency for burst shading.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------
    # 1️⃣ Plot raster
    # ------------------------------------------
    for ch, st in enumerate(spiketrain_list):
        spike_times = st.magnitude.flatten()

        if len(spike_times) == 0:
            continue

        y = np.ones_like(spike_times) * ch
        ax.plot(
            spike_times,
            y,
            spike_marker,
            markersize=spike_size,
            color="black",
            linestyle="None"
        )

    # ------------------------------------------
    # 2️⃣ Overlay network bursts
    # ------------------------------------------
    for start, end in network_bursts:
        ax.axvspan(
            start,
            end,
            color=burst_color,
            alpha=burst_alpha,
            linewidth=0
        )

    # ------------------------------------------
    # 3️⃣ Formatting
    # ------------------------------------------
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode")
    ax.set_title(title)

    ax.set_ylim(-1, len(spiketrain_list))
    ax.set_xlim(
        float(spiketrain_list[0].t_start),
        float(spiketrain_list[0].t_stop)
    )

    ax.invert_yaxis()   # optional; common in MEA raster plots
    ax.grid(False)

    plt.tight_layout()
    return fig

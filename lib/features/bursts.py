import numpy as np
import quantities as pq
from lib.features.helper import robust_stats
import matplotlib.pyplot as plt


def compute_burst_features(spiketrain_list, rec_dur_s=None, isi_threshold=0.1*pq.s, min_spikes_in_burst=3, min_burst_rate=0,flag_plot=False):
    
    if len(spiketrain_list) == 0:
        return {}
    
    if rec_dur_s is None:
        rec_dur_s = float((spiketrain_list[0].t_stop - spiketrain_list[0].t_start).rescale(pq.s))
    

    bursts = detect_bursts_for_all_channel(
        spiketrain_list,
        isi_threshold=isi_threshold,
        min_spikes_in_burst=min_spikes_in_burst
    )

    features = calc_features(
        bursts,
        rec_dur_s,
        min_burst_rate=min_burst_rate)

    fig = None
    if flag_plot:
        fig = plot_spiketrains_with_bursts(spiketrain_list, bursts)

    return features, fig


def calc_features(
    bursts_per_channel,
    rec_dur_s,
    min_burst_rate=0.05
):
    """
    Calculate chip-level burst statistics from detected bursts.

    Parameters
    ----------
    bursts_per_channel : dict
        Output of burst_features()
    rec_dur_s : float
        Recording duration in seconds
    min_burst_rate : float
        Minimum burst rate (Hz) to consider electrode active

    Returns
    -------
    features : dict
    """

    burst_rates = []
    burst_durations = []
    burst_ibis = []
    spikes_in_bursts = []
    active_flags = []

    for ch, bursts in bursts_per_channel.items():

        n_bursts = len(bursts)
        burst_rate = n_bursts / rec_dur_s

        is_active = burst_rate >= min_burst_rate
        active_flags.append(is_active)

        if not is_active or n_bursts == 0:
            burst_rates.append(np.nan)
            burst_durations.append(np.nan)
            burst_ibis.append(np.nan)
            spikes_in_bursts.append(np.nan)
            continue

        # durations (electrode-level median)
        durations = [b['duration'] for b in bursts]
        burst_durations.append(np.median(durations))

        # inter-burst interval
        ibis = [
            bursts[i]['start'] - bursts[i-1]['end']
            for i in range(1, len(bursts))
        ]
        burst_ibis.append(np.median(ibis) if len(ibis) > 0 else np.nan)

        # spikes in bursts
        sib = [b['spike_count'] for b in bursts]
        spikes_in_bursts.append(np.median(sib))

        burst_rates.append(burst_rate)

    # --------------------------------------------------
    # Aggregate to chip level
    # --------------------------------------------------
    features = {}

    active_flags = np.array(active_flags)
    features["ae"] = np.nanmean(active_flags.astype(float))

    features["rate_med"], features["rate_iqr"], features["rate_rcv"] = \
        robust_stats(burst_rates)

    features["dur_med"], features["dur_iqr"], features["dur_rcv"] = \
        robust_stats(burst_durations)

    features["ibi_med"], features["ibi_iqr"], features["ibi_rcv"] = \
        robust_stats(burst_ibis)

    features["sib_med"], features["sib_iqr"], features["sib_rcv"] = \
        robust_stats(spikes_in_bursts)

    return features


def detect_bursts_for_all_channel_median_isi(
    spiketrain_list,
    isi_threshold_factor=0.25,
    min_spikes_in_burst=3
):
    """
    Detect bursts for all electrodes.

    Parameters
    ----------
    spiketrain_list : list of neo.SpikeTrain
    isi_threshold_factor : float
    min_spikes_in_burst : int

    Returns
    -------
    bursts_per_channel : dict
        {
            channel_index: [
                {
                    'start': ...,
                    'end': ...,
                    'spike_count': ...,
                    'duration': ...
                }, ...
            ]
        }
    """

    bursts_per_channel = {}

    for ch, st in enumerate(spiketrain_list):

        spike_times = st.magnitude.flatten()

        bursts = _detect_bursts_for_one_channel_median_isi(
            spike_times,
            isi_threshold_factor=isi_threshold_factor,
            min_spikes_in_burst=min_spikes_in_burst
        )

        bursts_per_channel[ch] = bursts

    return bursts_per_channel


def detect_bursts_for_all_channel(
    spiketrain_list,
    isi_threshold=0.1*pq.s,
    min_spikes_in_burst=3
):
    """
    Detect bursts for all electrodes.

    Parameters
    ----------
    spiketrain_list : list of neo.SpikeTrain
    isi_threshold_factor : float
    min_spikes_in_burst : int

    Returns
    -------
    bursts_per_channel : dict
        {
            channel_index: [
                {
                    'start': ...,
                    'end': ...,
                    'spike_count': ...,
                    'duration': ...
                }, ...
            ]
        }
    """

    bursts_per_channel = {}

    for ch, st in enumerate(spiketrain_list):

        spike_times = st.magnitude.flatten()

        bursts = _detect_bursts_for_one_channel(
            spike_times,
            isi_threshold=isi_threshold,
            min_spikes_in_burst=min_spikes_in_burst
        )

        bursts_per_channel[ch] = bursts

    return bursts_per_channel


def _detect_bursts_for_one_channel_median_isi(spike_times, isi_threshold_factor=0.25, min_spikes_in_burst=3):
    """
    Detect bursts in a spike train using the ISI-threshold method.
    
    Returns a list of bursts, each as a dict:
        {
            'start': burst start time,
            'end': burst end time,
            'spike_count': number of spikes in the burst,
            'duration': end - start
        }
    """
    spike_times = np.sort(np.array(spike_times))
    n_spikes = len(spike_times)
    
    if n_spikes < min_spikes_in_burst:
        return []
    
    isis = np.diff(spike_times)
    median_isi = np.median(isis)
    isi_threshold = isi_threshold_factor * median_isi
    
    burst_start_indices = []
    burst_end_indices = []
    in_burst = False
    burst_start = 0

    for i, isi in enumerate(isis):
        if isi <= isi_threshold:
            if not in_burst:
                in_burst = True
                burst_start = i
        else:
            if in_burst:
                in_burst = False
                burst_end = i + 1
                if (burst_end - burst_start) >= min_spikes_in_burst:
                    burst_start_indices.append(burst_start)
                    burst_end_indices.append(burst_end)

    # Handle last burst
    if in_burst:
        burst_end = n_spikes
        if (burst_end - burst_start) >= min_spikes_in_burst:
            burst_start_indices.append(burst_start)
            burst_end_indices.append(burst_end)

    # Convert indices to bursts with spike count and duration
    bursts = []
    for start, end in zip(burst_start_indices, burst_end_indices):
        b_start = spike_times[start]
        b_end = spike_times[end-1]
        bursts.append({
            'start': b_start,
            'end': b_end,
            'spike_count': end - start,
            'duration': b_end - b_start
        })
    return bursts


def _detect_bursts_for_one_channel(spike_times, isi_threshold=0.1*pq.s, min_spikes_in_burst=3):
    """
    Detect bursts in a spike train using the ISI-threshold method.
    
    Returns a list of bursts, each as a dict:
        {
            'start': burst start time,
            'end': burst end time,
            'spike_count': number of spikes in the burst,
            'duration': end - start
        }
    """
    spike_times = np.sort(np.array(spike_times))
    n_spikes = len(spike_times)
    
    if n_spikes < min_spikes_in_burst:
        return []
    
    isis = np.diff(spike_times)

    burst_start_indices = []
    burst_end_indices = []
    in_burst = False
    burst_start = 0

    for i, isi in enumerate(isis):
        if isi <= isi_threshold:
            if not in_burst:
                in_burst = True
                burst_start = i
        else:
            if in_burst:
                in_burst = False
                burst_end = i + 1
                if (burst_end - burst_start) >= min_spikes_in_burst:
                    burst_start_indices.append(burst_start)
                    burst_end_indices.append(burst_end)

    # Handle last burst
    if in_burst:
        burst_end = n_spikes
        if (burst_end - burst_start) >= min_spikes_in_burst:
            burst_start_indices.append(burst_start)
            burst_end_indices.append(burst_end)

    # Convert indices to bursts with spike count and duration
    bursts = []
    for start, end in zip(burst_start_indices, burst_end_indices):
        b_start = spike_times[start]
        b_end = spike_times[end-1]
        bursts.append({
            'start': b_start,
            'end': b_end,
            'spike_count': end - start,
            'duration': b_end - b_start
        })
    return bursts


def plot_spiketrains_with_bursts(
    spiketrain_list,
    bursts_per_channel,
    figsize=(12, 6),
    spike_marker="|",
    spike_size=6,
    burst_color="tab:red",
    burst_alpha=0.35,
    title="Spike trains with detected bursts"
):
    """
    Plot raster of spike trains with single-channel bursts overlay.

    Parameters
    ----------
    spiketrain_list : list of neo.SpikeTrain
        One SpikeTrain per electrode.
    
    bursts_per_channel : dict or list
        Bursts for each channel.
        Expected format:
            {channel_index: [burst_dict, ...]}
        where each burst_dict:
            {
                'start': float,
                'end': float,
                'spike_count': int,
                'duration': float
            }

    figsize : tuple
        Figure size.
    spike_marker : str
        Marker used for spikes.
    spike_size : int
        Marker size.
    burst_color : str
        Color of burst bars.
    burst_alpha : float
        Transparency of burst bars.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # --------------------------------------------------
    # 1️⃣ Raster plot
    # --------------------------------------------------
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

    # --------------------------------------------------
    # 2️⃣ Plot bursts as horizontal bars
    # --------------------------------------------------
    for ch, bursts in bursts_per_channel.items():

        for b in bursts:
            start = b["start"]
            end = b["end"]

            #ax.hlines(
            #    y=ch,
            #    xmin=start,
            #    xmax=end,
            #    color=burst_color,
            #    linewidth=40,
            #    alpha=burst_alpha
            #)

            ax.add_patch(
                plt.Rectangle(
                    (start, ch - 0.35),
                    end - start,
                    0.7,
                    color=burst_color,
                    alpha=burst_alpha
                )
            )


    # --------------------------------------------------
    # 3️⃣ Formatting
    # --------------------------------------------------
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode")
    ax.set_title(title)

    ax.set_ylim(-1, len(spiketrain_list))
    ax.set_xlim(
        float(spiketrain_list[0].t_start),
        float(spiketrain_list[0].t_stop)
    )

    ax.invert_yaxis()  # common MEA raster style
    ax.grid(False)

    plt.tight_layout()
    return fig

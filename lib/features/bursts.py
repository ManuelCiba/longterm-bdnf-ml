import numpy as np
import quantities as pq
from lib.features.helper import robust_stats

# ----------------------------
# Burst detection for one spike train (ISI-threshold)
# ----------------------------
def detect_bursts_isi(spike_times, isi_threshold_factor=0.25, min_spikes_in_burst=3):
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


# ----------------------------
# 3️⃣ Burst features for list of electrodes (chip-level)
# ----------------------------
def compute_burst_features(spiketrain_list, rec_dur_s=None, isi_threshold_factor=0.25, min_spikes_in_burst=3, min_burst_rate=0.05):
    """
    Compute electrode-level burst features and aggregate to chip-level.

    Parameters
    ----------
    spiketrain_list : list of SpikeTrain (or array-like of spike times per electrode)
    rec_dur_s : float, optional
        Recording duration in seconds. If None, inferred from first SpikeTrain
    isi_threshold_factor : float
        Fraction of median ISI for burst detection
    min_spikes_in_burst : int
        Minimum spikes in burst
    min_burst_rate : float
        Minimum burst rate (Hz) to consider electrode active
    
    Returns
    -------
    chip_features : dict
        Aggregated burst features:
        - b_r_med, b_r_iqr, b_r_cvrobust
        - b_d_med, b_d_iqr, b_d_cvrobust
        - b_ibi_med, b_ibi_iqr, b_ibi_cvrobust
        - b_sib_med, b_sib_iqr, b_sib_cvrobust
        - b_ae : fraction of active burst electrodes
    """
    if len(spiketrain_list) == 0:
        return {}
    
    if rec_dur_s is None:
        rec_dur_s = float((spiketrain_list[0].t_stop - spiketrain_list[0].t_start).rescale(pq.s))
    
    # Lists to store electrode-level metrics
    burst_rates = []
    burst_durations = []
    burst_ibis = []
    spikes_in_bursts = []
    active_flags = []
    
    for st in spiketrain_list:
        spike_times = st.magnitude.flatten()  # seconds
        bursts = detect_bursts_isi(spike_times, isi_threshold_factor, min_spikes_in_burst)
        n_bursts = len(bursts)
        burst_rate = n_bursts / rec_dur_s
        
        # Active electrode criterion: min burst rate
        is_active = burst_rate >= min_burst_rate
        active_flags.append(is_active)
        
        if not is_active or n_bursts == 0:
            burst_rates.append(np.nan)
            burst_durations.append(np.nan)
            burst_ibis.append(np.nan)
            spikes_in_bursts.append(np.nan)
            continue
        
        
        
        
        # Burst durations
        durations = [b['duration'] for b in bursts]
        burst_durations.append(np.median(durations))  # per electrode median
        
        # Inter-burst intervals
        ibis = [bursts[i]['start'] - bursts[i-1]['end'] for i in range(1, len(bursts))]
        burst_ibis.append(np.median(ibis) if len(ibis) > 0 else np.nan)
        
        # Spikes in bursts
        sib = [b['spike_count'] for b in bursts]
        spikes_in_bursts.append(np.median(sib))
        
        # Burst rate
        burst_rates.append(burst_rate)
    
    # Aggregate to chip level
    chip_features = {}
    active_flags = np.array(active_flags)
    chip_features["b_ae"] = np.nanmean(active_flags.astype(float))
    
    chip_features["b_r_med"], chip_features["b_r_iqr"], chip_features["b_r_cvrobust"] = robust_stats(burst_rates)
    chip_features["b_d_med"], chip_features["b_d_iqr"], chip_features["b_d_cvrobust"] = robust_stats(burst_durations)
    chip_features["b_ibi_med"], chip_features["b_ibi_iqr"], chip_features["b_ibi_cvrobust"] = robust_stats(burst_ibis)
    chip_features["b_sib_med"], chip_features["b_sib_iqr"], chip_features["b_sib_cvrobust"] = robust_stats(spikes_in_bursts)
    
    return chip_features
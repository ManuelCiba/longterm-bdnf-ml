import numpy as np
import quantities as pq
from elephant.statistics import time_histogram
from lib.features.helper import robust_stats
from elephant.statistics import fanofactor

# ----------------------------
# Spike feature computation
# ----------------------------
def compute_spike_features(spiketrain_list, rec_dur_s=None, min_spike_rate=0.1):
    """
    Compute spike-level features for a list of Neo SpikeTrain objects (one per electrode)
    and aggregate across electrodes using median/IQR/CV-robust.
    
    Parameters
    ----------
    spiketrain_list : list of neo.SpikeTrain
        One SpikeTrain per electrode
    rec_dur_s : float, optional
        Recording duration in seconds. If None, use the first spiketrain's t_stop - t_start
    min_spike_rate : float
        Minimum spike rate (Hz) to consider electrode active
    
    Returns
    -------
    chip_features : dict
        Aggregated spike features at chip level:
        - spike_rate_med, spike_rate_iqr, spike_rate_cvrobust
        - fano_med, fano_iqr, fano_cvrobust
        - isi_med_med, isi_med_iqr, isi_med_cvrobust
        - isi_iqr_med, isi_iqr_iqr, isi_iqr_cvrobust
        - active_electrode_fraction
    """
    if len(spiketrain_list) == 0:
        return {}
    
    # Determine recording duration
    if rec_dur_s is None:
        rec_dur_s = float((spiketrain_list[0].t_stop - spiketrain_list[0].t_start).rescale(pq.s))
    
    # Lists to store electrode-level metrics
    spike_rates = []
    fano_factors = []
    isi_meds = []
    isi_iqrs = []
    active_flags = []
    amp_meds = []
    amp_iqrs = []
    
    # this fano factor is not used, it just shows how different the spike trains are between electrdoes 
    # the fano factor below is calculated at the electrdoe level, showing the variability of spike counts across time bins for each electrode, and then aggregated across electrodes using robust statistics.
    fano2 = fanofactor(spiketrain_list)

    for st in spiketrain_list:
        n_spikes = len(st)
        spike_rate = n_spikes / rec_dur_s
        is_active = spike_rate >= min_spike_rate
        active_flags.append(is_active)
        
        if not is_active:
            spike_rates.append(np.nan)
            fano_factors.append(np.nan)
            isi_meds.append(np.nan)
            isi_iqrs.append(np.nan)
            amp_meds.append(np.nan)
            amp_iqrs.append(np.nan)
            continue
        
        # Spike rate
        spike_rates.append(spike_rate)
        
        # Amplitudes
        amp = st.array_annotations.get("amplitude", np.nan)
        amp_med_val, amp_iqr_val, _ = robust_stats(np.abs(amp))
        amp_meds.append(amp_med_val)
        amp_iqrs.append(amp_iqr_val)
        
        # ISI
        isi = np.diff(st.magnitude).flatten()
        if len(isi) == 0:
            isi = np.array([np.nan])
        isi_med_val, isi_iqr_val, _ = robust_stats(isi)
        isi_meds.append(isi_med_val)
        isi_iqrs.append(isi_iqr_val)
        
        # Fano factor (1-s bins)
        if n_spikes < 2:
            fano = np.nan
        else:
            hist = time_histogram(st, binsize=1.0*pq.s, output='counts')
            fano = np.var(hist.magnitude) / np.mean(hist.magnitude) if np.mean(hist.magnitude) > 0 else np.nan
        fano_factors.append(fano)

        

    
    # ----------------------------
    # Aggregate across electrodes
    # ----------------------------
    chip_features = {}
    active_flags = np.array(active_flags)
    chip_features["active_electrodes_fraction"] = np.nanmean(active_flags.astype(float))
    chip_features["active_electrodes"] = np.sum(active_flags.astype(int))
    

    # Spike rate
    chip_features["spike_rate_med"], chip_features["spike_rate_iqr"], chip_features["spike_rate_cvrobust"] = \
        robust_stats(spike_rates)
    
    # Amp median
    chip_features["amplitude_med"], chip_features["amplitude_med_iqr"], chip_features["amplitude_med_cvrobust"] = \
        robust_stats(amp_meds)

    # Amp IQR
    chip_features["amplitude_iqr_med"], chip_features["amplitude_iqr_iqr"], chip_features["amplitude_iqr_cvrobust"] = \
        robust_stats(amp_iqrs)

    # Fano factor
    chip_features["fano_med"], chip_features["fano_iqr"], chip_features["fano_cvrobust"] = \
        robust_stats(fano_factors)
    
    # ISI median
    chip_features["isi_med_med"], chip_features["isi_med_iqr"], chip_features["isi_med_cvrobust"] = \
        robust_stats(isi_meds)
    
    # ISI IQR
    chip_features["isi_iqr_med"], chip_features["isi_iqr_iqr"], chip_features["isi_iqr_cvrobust"] = \
        robust_stats(isi_iqrs)
    
    return chip_features
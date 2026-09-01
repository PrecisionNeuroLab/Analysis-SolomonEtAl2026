def channel_lookup(loc, region, ch_names, column='Desikan_Killiany'):
# Create a function to look up bipolar channels by region
    import numpy as np
    
    region_rows = loc[loc[column].astype(str).str.contains(region)]
    labels = np.array(region_rows['FS_label'].str.lower())
    matched_bps = []
    for e in labels:
        for c in ch_names:
            if (e in c.lower().split('-')):
                matched_bps.append(c)
    matched_bps = np.unique(matched_bps)
    return [str(p) for p in matched_bps]  # ensures list is formatted correctly


def get_elec_dists(loc, chan_name,):
    import numpy as np
    import pandas as pd
    
    # Break down to monopolars for table lookup
    an = chan_name.split('-')[0]
    cath = chan_name.split('-')[1]
    
    # Get average of both monopolars to define a center for the radius
    an_data = loc[loc['FS_label'].str.upper()==str.upper(an)]
    cath_data = loc[loc['FS_label'].str.upper()==str.upper(cath)]
    avg_center = np.mean(np.array([an_data[['MNI_coord_1', 'MNI_coord_2', 'MNI_coord_3']], cath_data[['MNI_coord_1', 'MNI_coord_2', 'MNI_coord_3']] ]), 0)
    
    # Calculate distance to all other contacts
    elec_dists = []
    for i in range(len(loc)):
        mni_coords = np.array(loc.iloc[i][['MNI_coord_1', 'MNI_coord_2', 'MNI_coord_3']])
        dist = np.linalg.norm(avg_center-mni_coords)
        elec_dists.append(dist)

    # Note: returns locations in order of channels provided in localization structure, not MNE ch_names object. 
    return np.array(elec_dists)

def bipolarize_dists(bp_ch_names, elec_dists, loc):
    import numpy as np
    bp_dists = {}
    for bp_ch in bp_ch_names:
        if '-' in bp_ch:
            e1 = bp_ch.split('-')[0]
            e2 = bp_ch.split('-')[1]
            try:
                loc_idx1 = np.where(loc['FS_label'].str.upper()==e1.upper())[0][0]
                loc_idx2 = np.where(loc['FS_label'].str.upper()==e2.upper())[0][0]
            except:
                avg_dist = np.nan
            avg_dist = np.mean([elec_dists[loc_idx1], elec_dists[loc_idx2]])
            bp_dists[bp_ch] = avg_dist
        else:
            bp_dists[bp_ch] = np.nan
    return bp_dists
    


def construct_bipolar_montage(ch_names):
    '''
    ch_names is a list of channel names in string type. 
    '''
    import re
    import numpy as np
    import pandas as pd
    
    all_anodes = []; all_cathodes = [];

    # Identify channel groups
    group_labels = [re.sub('\d+', '', s) for s in ch_names]
    group_names = np.unique(group_labels)

    all_anodes = []
    all_cathodes = []

    # For each channel group, identify channels present and construct anode/cathode labels
    pd_names = pd.Series(ch_names)
    for g in group_names:
        group_chans = np.array(ch_names)[np.array(group_labels)==g]
        group_anodes = group_chans[0:-1]
        group_cathodes = group_chans[1:]

        all_anodes.extend(group_anodes)
        all_cathodes.extend(group_cathodes)

    return all_anodes, all_cathodes

def signalblender(a, b):
    """
    Blends two signals (should be equal legnth) by ramping down the amplitude of one as the other ramps up. 
    Should only do with short snippets to ensure relative stationarity.
    """
    import numpy as np
    
    a_weight = np.linspace(1, 0, num=a.size, endpoint=True)
    b_weight = np.linspace(0, 1, num=b.size, endpoint=True)
    
    # Apply weights to the signals
    a_weighted = a*a_weight
    b_weighted = b*b_weight
    
    # Combine the signals
    return a_weighted+b_weighted

def interpolate_stim(eeg, trigTimes, sfreq):
    """
    Uses signalblender to interpolate EEG during the stim artifact. Operates on one channel at a time.
    Scrubs -15ms to +15ms around stim artifact with 30ms blended, mirrored signals from the immediately preceeding and following EEG. 
    """
    from copy import copy
    eeg = copy(eeg)  # since we are modifying in-place
    
    for t in trigTimes: # for each stim pulse
        
        # Get the 50ms mirrored signals (buffered by 25 ms from the stim artifact itself)
        before = eeg[t-int(sfreq*0.045):t-int(sfreq*0.015)][::-1]
        after = eeg[t+int(sfreq*0.015):t+int(sfreq*0.045)][::-1]
        
        # Blend the signals
        blended = signalblender(before, after)
        
        # Replace the artifact interval
        eeg[t-int(sfreq*0.015):t+int(sfreq*0.015)] = blended
        
    return eeg

def assign_interval(step_id):
    if step_id in [1, 2]:
        interval = 'baseline'
    elif step_id in [4, 5]:
        interval = 'pre_TBS_1'
    elif step_id in [7]:
        interval = 'TBS_1'
    elif step_id in [8, 9]:
        interval = 'post_TBS_1'
    elif step_id in [17, 18]:
        interval = 'pre_TBS_2'
    elif step_id in [20]:
        interval = 'TBS_2'
    elif step_id in [21, 22]: 
        interval = 'post_TBS_2'
    elif step_id in [30, 31]:
        interval = 'pre_TBS_3'
    elif step_id in [33]:
        interval = 'TBS_3'
    elif step_id in [34, 35]:
        interval = 'post_TBS_3'
    elif step_id in [43, 44]:
        interval = 'pre_TBS_4'
    elif step_id in [46]:
        interval = 'TBS_4'
    elif step_id in [47, 48]:
        interval = 'post_TBS_4'
    else:
        interval = 'unknown'

    return interval


def process_subject_session(subject, tbs_block, params):
    """
    Processes a single subject and block. 
    Returns a list of dictionaries for all targets/ranges in that session.
    """
    import mne
    import numpy as np
    import pandas as pd
    from scipy.stats import zscore, ttest_ind
    
    # Unpack params
    all_ranges = params['all_ranges']
    freq_range = params['freq_range']
    baseline_range_ccep = params['baseline_range_ccep']
    num_network_channels = params['num_network_channels']
    outlier_threshold = params['outlier_threshold']
    
    local_results = []
    fn_prefix = f'../derivatives/pulse_labels/{subject}'
    
    try:
        # Load data once per worker task
        epochs_trigger = mne.read_epochs(f"{fn_prefix}_trigs-epo.fif", verbose=False).pick_types(seeg=True)
        loc = pd.read_csv(f'../data/{subject}/{subject}_loc.csv')
        
        meta = epochs_trigger.metadata
        all_targets = meta['stim_target'].unique()
        m_block = meta[meta['block'] == tbs_block]

        for targ in all_targets:
            m = m_block[m_block['stim_target'] == targ]
            if len(m) == 0: continue

            # Extract CCEP data
            pre_str = f'Pre{tbs_block}'
            post_str = f'Post{tbs_block}'
            epochs_ccep_pre = epochs_trigger[f'(block.str.contains("{pre_str}")) & (stim_target == "{targ}")']
            epochs_ccep_post = epochs_trigger[f'(block.str.contains("{post_str}")) & (stim_target == "{targ}")']

            # Exclude target lead
            exclude_chans, _ = exclude_target_lead(epochs_ccep_pre)
            epochs_ccep_pre.info['bads'] += exclude_chans
            epochs_ccep_post.info['bads'] += exclude_chans

            # Z-score and Rank
            epochs_z = epochs_ccep_pre.copy().apply_function(zscore_epoch, times=epochs_ccep_pre.times, baseline=(-0.2, 0))
            evoked_z = epochs_z.average()
            evoked_z.drop_channels(evoked_z.info['bads'])
            
            df_ptp, top_chans, bottom_chans = rank_channels_by_ptp(
                evoked_z, n_top=num_network_channels, n_bottom=num_network_channels
            )
            
            channels = [s.upper() for s in evoked_z.ch_names]

            for ccep_range in all_ranges:
                spec_pre = get_ccep_power(epochs_ccep_pre, freq_range, baseline_range_ccep, ccep_range)
                spec_post = get_ccep_power(epochs_ccep_post, freq_range, baseline_range_ccep, ccep_range)

                # Outlier removal
                spec_pre[np.abs(zscore(spec_pre, axis=None, nan_policy='omit')) > outlier_threshold] = np.nan
                spec_post[np.abs(zscore(spec_post, axis=None, nan_policy='omit')) > outlier_threshold] = np.nan

                # Stats
                ccep_diff, _ = ttest_ind(spec_post, spec_pre, equal_var=False, nan_policy='omit')
                
                top_idxs = [channels.index(c.upper()) for c in top_chans]
                bottom_idxs = [channels.index(c.upper()) for c in bottom_chans]
                
                local_results.append({
                    'subject': subject,
                    'block': tbs_block,
                    'target': targ,
                    'range': ccep_range,
                    'entrained_pwr': np.nanmean(ccep_diff[top_idxs]),
                    'unentrained_pwr': np.nanmean(ccep_diff[bottom_idxs])
                })
                
    except Exception as e:
        print(f"Error processing {subject} {tbs_block}: {e}")
        
    return local_results
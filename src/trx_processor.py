import multiprocessing as mp
import os
from datetime import datetime
import h5py
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.style.use('../anemotaxis.mplstyle')

def get_behavior_data(f, field, i):
    """Extract behavior-related cell arrays from MATLAB struct.
    
    Args:
        f: HDF5 file object
        field: Field name ('duration_large', 't_start_stop_large', or 'nb_action_large')
        i: Larva index
        
    Returns:
        list: List of 7 arrays, one for each behavior type
            duration_large: List of 1D arrays with durations of each behavior event
            t_start_stop_large: List of 2D arrays with start/stop times of each event
            nb_action_large: List of counts of events per behavior type
    """
    try:
        # Get the reference to the cell array
        cell_ref = f['trx'][field][0][i]
        if not isinstance(cell_ref, h5py.h5r.Reference):
            return None
            
        # Get the 1x7 cell array
        cell_array = f[cell_ref]
        
        # Initialize list to store arrays for each behavior
        behavior_data = []
        
        # Extract each behavior's data (7 or 12 behaviors total)
        for j in range(7):
            try:
                # Get reference to data for this behavior
                behavior_ref = cell_array[j,0]  # MATLAB stores in column-major order
                if isinstance(behavior_ref, h5py.h5r.Reference):
                    behavior_array = f[behavior_ref]
                    behavior_data.append(np.array(behavior_array))
                else:
                    behavior_data.append(np.array([]))
            except Exception as e:
                tqdm.write(f"Error getting behavior {j+1} data: {str(e)}")
                behavior_data.append(np.array([]))
        
        return behavior_data
        
    except Exception as e:
        tqdm.write(f"Error getting behavior data for field {field}: {str(e)}")
        return None

def process_single_file(file_path, show_progress=False):
    """Process a single trx.mat file containing larval tracking data.
    
    This function extracts behavioral and tracking data from a MATLAB-generated trx.mat file.
    The file contains tracking information for multiple larvae including:
    - Time series data
    - Spine and contour coordinates 
    - Body part positions (head, tail, etc.)
    - Behavioral state information
    
    Shape information for key arrays:
    - t: (steps,) - Time points for each frame
    - x_spine, y_spine: (steps, 11) - 11 points along the larva's spine
    - x_contour, y_contour: (steps, 500) - 500 points defining the larva's outline
    - x_center, y_center, etc.: (steps,) - Track points for body parts
    - global_state_large_state: (steps,) - Behavioral state per frame
        States: 1=run, 2=cast, 3=stop, 4=hunch, 5=backup, 6=roll, 7=small actions
    - t_start_stop_large: List of 7 arrays - Start/stop times for each behavior type
    - duration_large: List of 7 arrays - Duration of each behavior type
    - nb_action_large: List of 7 arrays - Number of events per behavior type
    
    Args:
        file_path (str): Path to the trx.mat file
        show_progress (bool): Whether to show progress messages
        
    Returns:
        tuple: (date_str, extracted_data, metadata)
            - date_str (str): Experiment date from folder name
            - extracted_data (dict): Dictionary of larva_id -> tracking data
            - metadata (dict): File info including path, date, number of larvae
    """
    try:
        with h5py.File(file_path, 'r') as f:
            fields = list(f['trx'].keys())
            nb_larvae = f['trx'][fields[0]].shape[1]
            
            if show_progress:
                print(f"\nProcessing file: {file_path}")
                print(f"Number of larvae: {nb_larvae}")
            
            extracted_data = {}
            for i in tqdm(range(nb_larvae), desc="Processing larvae"):
                larva = {}
                try:
                    # Helper function to safely extract array data
                    def get_array(field):
                        ref = f['trx'][field][0][i]
                        if isinstance(ref, h5py.Dataset):
                            return np.array(ref)
                        return np.array(f[ref])
                    
                    # Time series data
                    larva['t'] = get_array('t')  # (steps,)

                    # Head & Tail Velocity data
                    larva['head_velocity_norm_smooth_5'] = get_array('head_velocity_norm_smooth_5')  # (steps,)
                    larva['tail_velocity_norm_smooth_5'] = get_array('tail_velocity_norm_smooth_5')  # (steps,)
                    larva['angle_upper_lower_smooth_5'] = get_array('angle_upper_lower_smooth_5')  # (steps,)
                    larva['angle_downer_upper_smooth_5'] = get_array('angle_downer_upper_smooth_5')  # (steps,)
                    
                    # Spine coordinates
                    larva['x_spine'] = get_array('x_spine')  # (steps, 11)
                    larva['y_spine'] = get_array('y_spine')  # (steps, 11)
                    
                    # Contour coordinates  
                    larva['x_contour'] = get_array('x_contour')  # (steps, 500)
                    larva['y_contour'] = get_array('y_contour')  # (steps, 500)
                    
                    # Single point tracking coordinates
                    for point in ['center', 'neck', 'head', 'tail', 'neck_down', 'neck_top']:
                        larva[f'x_{point}'] = get_array(f'x_{point}')  # (steps,)
                        larva[f'y_{point}'] = get_array(f'y_{point}')  # (steps,)
                    
                    # Behavioral state data
                    larva['global_state_large_state'] = get_array('global_state_large_state')
                    larva['global_state_small_large_state'] = get_array('global_state_small_large_state')
                    
                    # Get behavior metrics - each will be list of 7 arrays
                    larva['duration_large'] = get_behavior_data(f, 'duration_large', i)
                    larva['t_start_stop_large'] = get_behavior_data(f, 't_start_stop_large', i)
                    larva['start_stop_large'] = get_behavior_data(f, 'start_stop_large', i)
                    larva['nb_action_large'] = get_behavior_data(f, 'nb_action_large', i)

                    # Get behavior metrics - each will be list of 12 arrays
                    larva['duration_large_small'] = get_behavior_data(f, 'duration_large_small', i)
                    larva['t_start_stop_large_small'] = get_behavior_data(f, 't_start_stop_large_small', i)
                    larva['start_stop_large_small'] = get_behavior_data(f, 'start_stop_large_small', i)
                    larva['nb_action_large_small'] = get_behavior_data(f, 'nb_action_large_small', i)
                    
                    # Get larva ID
                    larva_id_ref = f['trx']['numero_larva_num'][0][i]
                    if isinstance(larva_id_ref, h5py.Dataset):
                        larva_id = int(np.array(larva_id_ref))
                    else:
                        larva_id = int(np.array(f[larva_id_ref]))
                    
                    # Only add larva if we have valid behavior data
                    if all(x is not None for x in [
                        larva['duration_large'],
                        larva['t_start_stop_large'],
                        larva['nb_action_large']
                    ]):
                        extracted_data[larva_id] = larva
                    
                except Exception as e:
                    tqdm.write(f"Error extracting data for larva {i}: {str(e)}")
                    continue
            
            date_str = os.path.basename(os.path.dirname(file_path))
            
            return date_str, extracted_data, {
                'path': file_path,
                'date': datetime.strptime(date_str.split('_')[0], '%Y%m%d'),
                'n_larvae': len(extracted_data),
                'date_str': date_str
            }
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_all_trx_files(base_path):
    """Process all trx.mat files in a directory tree sequentially with progress bar."""
    
    # Find all trx files
    file_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'trx.mat':
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    
    print(f"Found {len(file_list)} trx.mat files")
    
    # Initialize results
    all_data = {}
    metadata = {
        'files_processed': [],
        'total_larvae': 0,
        'experiments': {}
    }
    
    # Process files sequentially with progress bar
    for file_path in tqdm(file_list, desc="Processing trx files"):
        result = process_single_file(file_path)
        if result is not None:
            date_str, trx_extracted, exp_info = result
            metadata['experiments'][date_str] = exp_info
            metadata['files_processed'].append(exp_info['path'])
            metadata['total_larvae'] += exp_info['n_larvae']
            
            for larva_id, larva_data in trx_extracted.items():
                unique_id = f"{date_str}_{larva_id}"
                larva_data['experiment_date'] = date_str
                all_data[unique_id] = larva_data
    
    print(f"\nProcessed {len(metadata['files_processed'])} files")
    print(f"Total larvae: {metadata['total_larvae']}")
    
    return {'data': all_data, 'metadata': metadata}

def process_all_trx_files_parallel(base_path, n_processes=None):

    """Process all trx.mat files in a directory tree using parallel processing."""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    # Find all trx files
    file_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'trx.mat':
                file_path = os.path.join(root, file)
                file_list.append(file_path)  # Only append the path, not a tuple
    
    print(f"Found {len(file_list)} trx.mat files")
    print(f"Using {n_processes} processes")
    
    # Initialize results
    all_data = {}
    metadata = {
        'files_processed': [],
        'total_larvae': 0,
        'experiments': {}
    }
    
    # Process files in parallel
    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, file_list),
            total=len(file_list),
            desc="Processing files"
        ))
    
    # Combine results
    for result in results:
        if result is not None:
            date_str, trx_extracted, exp_info = result
            metadata['experiments'][date_str] = exp_info
            metadata['files_processed'].append(exp_info['path'])
            metadata['total_larvae'] += exp_info['n_larvae']
            
            for larva_id, larva_data in trx_extracted.items():
                unique_id = f"{date_str}_{larva_id}"
                larva_data['experiment_date'] = date_str
                all_data[unique_id] = larva_data
    
    print(f"\nProcessed {len(metadata['files_processed'])} files")
    print(f"Total larvae: {metadata['total_larvae']}")
    
    return {'data': all_data, 'metadata': metadata}

def filter_larvae_by_duration(data, min_total_duration=None, percentile=10):
    """Filter larvae based on their total tracked duration.
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        min_total_duration: Minimum total duration in seconds. If None, uses percentile
        percentile: Percentile threshold (0-100) to use if min_total_duration is None
        
    Returns:
        dict: Filtered data with same structure as input, excluding larvae below threshold
    """
    # Handle data type
    if 'data' in data:
        extracted_data = data['data']
        is_multi_exp = True
    else:
        extracted_data = data
        is_multi_exp = False
    
    # Calculate total duration for each larva
    larva_durations = {}
    for larva_id, larva_data in extracted_data.items():
        total = 0
        for durations in larva_data['duration_large']:
            if durations is not None:
                total += float(np.nansum(durations.flatten()))
        larva_durations[larva_id] = total
    
    # Determine threshold
    if min_total_duration is None:
        min_total_duration = np.percentile(list(larva_durations.values()), percentile)
    
    # Filter larvae
    filtered_data = {}
    for larva_id, total_duration in larva_durations.items():
        if total_duration >= min_total_duration:
            filtered_data[larva_id] = extracted_data[larva_id]
    
    # Return with appropriate structure
    if is_multi_exp:
        return {
            'data': filtered_data,
            'metadata': {
                **data['metadata'],
                'total_larvae': len(filtered_data),
                'duration_threshold': min_total_duration,
                'original_total_larvae': len(extracted_data)
            }
        }
    else:
        return filtered_data

def analyze_behavior_durations(data, show_plot=True, title=None):
    """Analyze behavior durations from processed trx data with separate analysis for large, small, and total behaviors.
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        show_plot (bool): Whether to show visualization
        title (str): Optional plot title override
        
    Returns:
        dict: Statistics for each behavior type (large, small, total)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Define behavior mappings
    large_behavior_names = {
        1: 'run',
        2: 'cast',
        3: 'stop',
        4: 'hunch',
        5: 'backup',
        6: 'roll'
        # Removed 7: 'small_actions'
    }
    
    small_behavior_names = {
        0.5: 'small_run',
        1.5: 'small_cast',
        2.5: 'small_stop',
        3.5: 'small_hunch',
        4.5: 'small_backup',
        5.5: 'small_roll'
        # Removed 6.5: 'small_small_actions'
    }
    
    # Define color scheme to match global behavior matrix
    behavior_colors = {
        'run': [0.0, 0.0, 0.0],      # Black (for Crawl)
        'cast': [1.0, 0.0, 0.0],     # Red (for Bend)
        'stop': [0.0, 1.0, 0.0],     # Green (for Stop)
        'hunch': [0.0, 0.0, 1.0],    # Blue (for Hunch)
        'backup': [1.0, 0.5, 0.0],   # Orange
        'roll': [0.5, 0.0, 0.5]      # Purple
        # Removed 'small_actions': [0.7, 0.7, 0.7]  # Light gray
    }
    
    # Combined behavior mapping for display
    all_behavior_names = large_behavior_names.copy()
    all_behavior_names.update(small_behavior_names)
    
    # Handle data type
    if 'data' in data:
        extracted_data = data['data']
        n_larvae = data['metadata']['total_larvae']
        if title is None:
            title = 'All Experiments'
    else:
        extracted_data = data
        n_larvae = len(extracted_data)
        if title is None:
            title = 'Single Experiment'
    
    # Initialize statistics for each behavior group
    behavior_stats = {}
    
    # Initialize large behaviors
    for name in large_behavior_names.values():
        behavior_stats[f"{name}_large"] = {
            'durations': [],
            'n_actions': 0,
            'total_duration': 0,
            'mean_duration': 0,
            'std_duration': 0,
            'percent_of_total': 0,
            'per_larva_total': []  # Total duration per larva
        }
    
    # Initialize small behaviors
    for name in small_behavior_names.values():
        behavior_stats[name] = {
            'durations': [],
            'n_actions': 0,
            'total_duration': 0,
            'mean_duration': 0,
            'std_duration': 0,
            'percent_of_total': 0,
            'per_larva_total': []  # Total duration per larva
        }
    
    # Initialize total behaviors (combined large and small)
    for base_name in ['run', 'cast', 'stop', 'hunch', 'backup', 'roll']:
        behavior_stats[f"{base_name}_total"] = {
            'durations': [],
            'n_actions': 0,
            'total_duration': 0,
            'mean_duration': 0,
            'std_duration': 0,
            'percent_of_total': 0,
            'per_larva_total': []  # Total duration per larva
        }
    
    # Process each larva
    total_actions = {'large': 0, 'small': 0, 'total': 0}
    
    for larva_id, larva_data in extracted_data.items():
        # Initialize per-larva totals
        larva_totals_large = {f"{b}_large": 0 for b in large_behavior_names.values()}
        larva_totals_small = {b: 0 for b in small_behavior_names.values()}
        larva_totals_combined = {f"{b}_total": 0 for b in large_behavior_names.values()}
        
        # Count actions and get durations for LARGE behaviors
        for state_idx, (n_actions, durations) in enumerate(zip(
            larva_data['nb_action_large'], 
            larva_data['duration_large']
        )):
            if n_actions is not None and durations is not None:
                # Skip if behavior is not in our mapping (e.g., 7: small_actions)
                behavior_id = state_idx + 1
                if behavior_id not in large_behavior_names:
                    continue
                    
                # Clean data - remove NaN values
                n = int(np.nansum(n_actions.flatten()))  # Use nansum for action counts
                clean_durations = durations.flatten()[~np.isnan(durations.flatten())]
                
                if n > 0 and len(clean_durations) > 0:
                    behavior_name = large_behavior_names[behavior_id]
                    
                    # Add to large behavior stats
                    large_key = f"{behavior_name}_large"
                    behavior_stats[large_key]['n_actions'] += n
                    behavior_stats[large_key]['durations'].extend(clean_durations)
                    larva_totals_large[large_key] += float(np.nansum(clean_durations))
                    
                    # Add to total behavior stats
                    total_key = f"{behavior_name}_total"
                    behavior_stats[total_key]['n_actions'] += n
                    behavior_stats[total_key]['durations'].extend(clean_durations)
                    larva_totals_combined[total_key] += float(np.nansum(clean_durations))
                    
                    total_actions['large'] += n
                    total_actions['total'] += n
        
        # Process SMALL behaviors from large_small array (which has 12 elements)
        if 'duration_large_small' in larva_data and 'nb_action_large_small' in larva_data:
            # Get the large-small arrays
            duration_large_small = larva_data['duration_large_small']
            nb_action_large_small = larva_data['nb_action_large_small']
            
            # Process only small behaviors (indices 1, 3, 5, 7, 9, 11 in the 12-element array)
            for i, small_state in enumerate([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]):
                # Map small state to its index in the array (odd indices)
                idx = i * 2 + 1
                
                # Check if index is valid for the array
                if idx < len(duration_large_small) and idx < len(nb_action_large_small):
                    durations = duration_large_small[idx]
                    n_actions = nb_action_large_small[idx]
                    
                    if durations is not None and n_actions is not None:
                        # Clean data - remove NaN values
                        n = int(np.nansum(n_actions.flatten()))
                        clean_durations = durations.flatten()[~np.isnan(durations.flatten())]
                        
                        if n > 0 and len(clean_durations) > 0:
                            small_behavior_name = small_behavior_names[small_state]
                            
                            # Add to small behavior stats
                            behavior_stats[small_behavior_name]['n_actions'] += n
                            behavior_stats[small_behavior_name]['durations'].extend(clean_durations)
                            larva_totals_small[small_behavior_name] += float(np.nansum(clean_durations))
                            
                            # Get the base behavior name by removing "small_" prefix
                            base_behavior = small_behavior_name.replace('small_', '')
                            total_key = f"{base_behavior}_total"
                            
                            # Add to total behavior stats
                            behavior_stats[total_key]['n_actions'] += n
                            behavior_stats[total_key]['durations'].extend(clean_durations)
                            larva_totals_combined[total_key] += float(np.nansum(clean_durations))
                            
                            total_actions['small'] += n
                            total_actions['total'] += n
        
        # Store per-larva totals
        for key, total in larva_totals_large.items():
            behavior_stats[key]['per_larva_total'].append(total)
            
        for key, total in larva_totals_small.items():
            behavior_stats[key]['per_larva_total'].append(total)
            
        for key, total in larva_totals_combined.items():
            behavior_stats[key]['per_larva_total'].append(total)
    
    # Calculate statistics using nan-aware functions
    for behavior in behavior_stats:
        stats = behavior_stats[behavior]
        if stats['durations']:
            # Determine the behavior group
            if '_large' in behavior:
                group_total = total_actions['large']
            elif behavior in small_behavior_names.values():
                group_total = total_actions['small']
            else:  # This is a _total behavior
                group_total = total_actions['total']
                
            stats.update({
                'total_duration': float(np.nansum(stats['durations'])),
                'mean_duration': float(np.nanmean(stats['durations'])),
                'std_duration': float(np.nanstd(stats['durations'])),
                'percent_of_total': 100 * stats['n_actions'] / group_total if group_total > 0 else 0
            })
    
    if show_plot:
        # Create plots for each behavior group (large, small, total)
        for behavior_group in ['large', 'small', 'total']:
            # Skip small if no small data
            if behavior_group == 'small' and total_actions['small'] == 0:
                continue
            
            # Create filtered list of behaviors for this group
            if behavior_group == 'large':
                behaviors = [f"{b}_large" for b in large_behavior_names.values()]
                group_title = f"{title} - Large Behaviors"
            elif behavior_group == 'small':
                behaviors = list(small_behavior_names.values())
                group_title = f"{title} - Small Behaviors"
            else:  # total
                behaviors = [f"{b}_total" for b in large_behavior_names.values()]
                group_title = f"{title} - Total Behaviors"
                
            # Filter out behaviors with no events
            behaviors = [b for b in behaviors if behavior_stats[b]['n_actions'] > 0]
            
            if not behaviors:  # Skip if no behaviors in this group
                continue
                
            # Get positions and labels
            positions = np.arange(len(behaviors))
            labels = []
            for b in behaviors:
                # Extract base behavior name
                base_name = b.replace('_large', '').replace('_total', '')
                if base_name.startswith('small_'):
                    base_name = base_name[6:]  # Remove 'small_' prefix
                
                # Create label
                label = f"{base_name}\n{behavior_stats[b]['n_actions']} events\n({behavior_stats[b]['percent_of_total']:.1f}%)"
                labels.append(label)
            
            # Create figure with shared x-axis
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.1)
            
            # Create subplots with shared x-axis
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            
            # First plot: Event durations violin plot
            valid_data = []
            valid_positions = []
            valid_colors = []
            
            for idx, behavior in enumerate(behaviors):
                durations = behavior_stats[behavior]['durations']
                if len(durations) > 0:
                    valid_data.append(durations)
                    valid_positions.append(positions[idx])
                    
                    # Get base behavior name for color lookup
                    base_name = behavior.replace('_large', '').replace('_total', '')
                    if base_name.startswith('small_'):
                        base_name = base_name[6:]  # Remove 'small_' prefix
                    
                    valid_colors.append(behavior_colors[base_name])
            
            if valid_data:
                violin_parts = ax1.violinplot(valid_data, 
                                            positions=valid_positions, 
                                            showmeans=False,
                                            showextrema=False)
                
                # Color each violin according to behavior
                for body, color in zip(violin_parts['bodies'], valid_colors):
                    body.set_alpha(0.6)
                    body.set_facecolor(color)
                    body.set_edgecolor(color)
                
            # Add quartiles with enhanced visibility
            for idx, behavior in enumerate(behaviors):
                durations = behavior_stats[behavior]['durations']
                if len(durations) > 0:
                    quartiles = np.percentile(durations, [25, 50, 75])
                    
                    # Get base behavior name for color lookup
                    base_name = behavior.replace('_large', '').replace('_total', '')
                    if base_name.startswith('small_'):
                        base_name = base_name[6:]  # Remove 'small_' prefix
                    
                    color = behavior_colors[base_name]
                    
                    # Vertical line for IQR range
                    ax1.vlines(positions[idx], quartiles[0], quartiles[2], 
                            color=color, lw=1.5)
                    
                    # Horizontal lines at each quartile
                    for q in quartiles:
                        ax1.hlines(q, positions[idx]-0.15, positions[idx]+0.15,
                                color=color, lw=1.5)
                    
            ax1.set_title('Event Durations Distribution')
            ax1.set_ylabel('Duration (seconds)')
            
            # Second plot: Total duration violin plot
            total_violin = ax2.violinplot([behavior_stats[b]['per_larva_total'] for b in behaviors],
                                        positions=positions,
                                        showmeans=False,
                                        showextrema=False)
            
            # Color total duration violins
            for idx, body in enumerate(total_violin['bodies']):
                # Get base behavior name for color lookup
                base_name = behaviors[idx].replace('_large', '').replace('_total', '')
                if base_name.startswith('small_'):
                    base_name = base_name[6:]  # Remove 'small_' prefix
                
                color = behavior_colors[base_name]
                body.set_alpha(0.6)
                body.set_facecolor(color)
                body.set_edgecolor(color)
            
            # Add quartiles for total durations with enhanced visibility
            for idx, behavior in enumerate(behaviors):
                totals = behavior_stats[behavior]['per_larva_total']
                if len(totals) > 0:
                    quartiles = np.percentile(totals, [25, 50, 75])
                    
                    # Get base behavior name for color lookup
                    base_name = behavior.replace('_large', '').replace('_total', '')
                    if base_name.startswith('small_'):
                        base_name = base_name[6:]  # Remove 'small_' prefix
                    
                    color = behavior_colors[base_name]
                    
                    # Vertical line for IQR range with same color as behavior
                    ax2.vlines(positions[idx], quartiles[0], quartiles[2], 
                            color=color, lw=1.5)
                    
                    # Horizontal lines for all quartiles with same color and width
                    for q in quartiles:
                        ax2.hlines(q, positions[idx]-0.15, positions[idx]+0.15,
                                color=color, lw=1.5)
            
            ax2.set_ylabel('Total Duration (s)')
            
            # Third plot: Action counts
            bars = ax3.bar(positions, 
                          [behavior_stats[b]['n_actions'] for b in behaviors],
                          color=[behavior_colors[b.replace('_large', '').replace('_total', '')
                                              .replace('small_', '', 1)] 
                                for b in behaviors],
                          alpha=0.6)
            
            # Add percentage labels
            for idx, bar in enumerate(bars):
                height = bar.get_height()
                pct = behavior_stats[behaviors[idx]]['percent_of_total']
                ax3.text(bar.get_x() + bar.get_width()/2, height,
                        f'{pct:.1f}%',
                        ha='center', va='bottom')
            
            ax3.set_ylabel('Number of Events')
            
            # Hide x-axis for top two plots
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            ax1.set_xlabel('')
            ax2.set_xlabel('')
            
            # Show only bottom x-axis
            ax3.set_xticks(positions)
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            
            # Remove top spines
            for ax in [ax1, ax2, ax3]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            fig.suptitle(f"{group_title}\n(n = {n_larvae} larvae)", y=1.02)
            plt.tight_layout()
            plt.show()
    
    # Print summary statistics for each group
    for group_name, group in [
        ('Large Behaviors', '_large'), 
        ('Small Behaviors', 'small_'),
        ('Total Behaviors', '_total')
    ]:
        # Get relevant behaviors for this group
        if group == '_large':
            behaviors = [f"{b}_large" for b in large_behavior_names.values()]
        elif group == 'small_':
            behaviors = list(small_behavior_names.values())
        else:  # _total
            behaviors = [f"{b}_total" for b in large_behavior_names.values()]
            
        # Filter behaviors with data
        behaviors = [b for b in behaviors if behavior_stats[b]['durations']]
        
        if behaviors:
            print(f"\n{group_name} analysis for {title}")
            print(f"Number of larvae: {n_larvae}")
            print(f"Total actions: {total_actions.get(group_name.lower().split()[0], 0)}\n")
            print(f"{'Behavior':>12} {'Events':>8} {'%Total':>7} {'Mean(s)':>12} {'Median(s)':>10}")
            print("-" * 60)
            
            for behavior in behaviors:
                stats = behavior_stats[behavior]
                if stats['durations']:
                    median = float(np.median(stats['durations']))
                    
                    # Get readable name
                    display_name = behavior.replace('_large', '').replace('_total', '')
                    
                    print(f"{display_name:>12}: {stats['n_actions']:8d} {stats['percent_of_total']:6.1f}%"
                          f"{stats['mean_duration']:10.2f} {median:10.2f}")
    
    return behavior_stats

def plot_global_behavior_matrix(trx_data):
    """
    Plot global behavior using the global state.
    Colors frames based on global_state_large_state value with distinct colors:
    1 -> light red (run)
    2 -> light blue (cast)
    3 -> light green (stop)
    4 -> light purple (head cast)
    5 -> yellow (backup)
    6 -> orange (roll)
    7 -> pink (small actions)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Get sorted larva IDs
    larva_ids = sorted(trx_data.keys())
    n_larvae = len(larva_ids)
    
    # Compute time range
    tmins = [np.min(trx_data[lid]['t']) for lid in larva_ids if len(trx_data[lid]['t']) > 0]
    tmaxs = [np.max(trx_data[lid]['t']) for lid in larva_ids if len(trx_data[lid]['t']) > 0]
    if not tmins or not tmaxs:
        raise ValueError("No time data found!")
    min_time = float(min(tmins))
    max_time = float(max(tmaxs))
    
    resolution = 1000
    behavior_matrix = np.full((n_larvae, resolution, 3), fill_value=1.0)  # white background

    # Define colors for each state
    state_colors = {
        1: [0.0, 0.0, 0.0],      # Black (for Crawl)
        2: [1.0, 0.0, 0.0],     # Red (for Bend)
        3: [0.0, 1.0, 0.0],     # Green (for Stop)
        4: [0.0, 0.0, 1.0],    # Blue (for Hunch)
        5: [1.0, 0.5, 0.0],   # Orange
        6: [0.5, 0.0, 0.5],     # Purple
        7: [0.7, 0.7, 0.7]  # Light gray (for Small action)
    }

    # Process each larva
    for i, lid in enumerate(larva_ids):
        larva_time = np.array(trx_data[lid]['t']).flatten()
        states = np.array(trx_data[lid]['global_state_large_state']).flatten()
        
        # Convert times to indices
        time_indices = np.floor(
            ((larva_time - min_time) / (max_time - min_time) * (resolution - 1))
        ).astype(int)
        time_indices = np.clip(time_indices, 0, resolution - 1)

        # For each unique time index, use the corresponding state
        unique_indices = np.unique(time_indices)
        for t_idx in unique_indices:
            mask = time_indices == t_idx
            state = int(states[mask][0])  # Take first state if multiple exist
            
            # Assign color based on state
            behavior_matrix[i, t_idx] = state_colors.get(state, [1, 1, 1])  # Default to white if state not found

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(behavior_matrix, aspect='auto', interpolation='nearest', alpha=0.8,
               extent=[min_time, max_time, n_larvae, 0])
    plt.yticks(np.arange(0.5, n_larvae + 0.5), larva_ids, fontsize='small')
    plt.title('Global Behavior States', pad=20)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Larva ID')
    
    # Add legend with all states
    legend_elements = [
        Patch(facecolor=state_colors[1], label='Run'),
        Patch(facecolor=state_colors[2], label='Cast'),
        Patch(facecolor=state_colors[3], label='Stop'),
        Patch(facecolor=state_colors[4], label='Head Cast'),
        Patch(facecolor=state_colors[5], label='Backup'),
        Patch(facecolor=state_colors[6], label='Roll'),
        Patch(facecolor=state_colors[7], label='Small Actions'),
        Patch(facecolor=[1, 1, 1], label='Other')
    ]
    plt.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5), title='Behavioral States')
    plt.tight_layout()
    plt.show()
    
    return behavior_matrix

def plot_behavioral_contour_with_global_trajectory(trx_data, larva_id):
    """
    Plot a filled larva contour with spine and neck points, its global trajectory,
    and angle dynamics over time with behavior state coloring.
    
    Colors based on global_state_large_state value:
    1 -> Black (run/crawl)
    2 -> Red (cast/bend)
    3 -> Green (stop)
    4 -> Blue (hunch)
    5 -> Orange (backup)
    6 -> Purple (roll)
    7 -> Light gray (small actions)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from ipywidgets import Play, IntSlider, HBox, jslink
    from IPython.display import display
    from matplotlib.patches import Patch

    # Define visualization parameters
    WINDOW_SIZE = 2  # Size of zoom window
    FIGURE_SIZE = (12, 6)  # Original size
    LINE_WIDTH = 2
    MARKER_SIZE = 8
    SPINE_MARKER_SIZE = 4  # Smaller marker size for spine points
    ALPHA = 0.6
    TIME_WINDOW = 50  # Show 50 seconds of angle data

    # Get data and ensure proper shapes
    larva = trx_data[larva_id]
    x_contour = np.atleast_2d(np.array(larva['x_contour']))
    y_contour = np.atleast_2d(np.array(larva['y_contour']))
    x_center = np.array(larva['x_center']).flatten()
    y_center = np.array(larva['y_center']).flatten()
    x_spine = np.array(larva['x_spine'])
    y_spine = np.array(larva['y_spine'])
    x_neck = np.array(larva['x_neck']).flatten()
    y_neck = np.array(larva['y_neck']).flatten()
    time = np.array(larva['t']).flatten()
    global_state = np.array(larva['global_state_large_state'])
    
    # Get the angle data and convert to degrees
    try:
        angle_downer_upper = np.array(larva['angle_downer_upper_smooth_5']).flatten()
        # Convert to degrees and reverse
        angle_downer_upper_deg = -1 * np.degrees(angle_downer_upper)
    except KeyError:
        print("Warning: angle_downer_upper_smooth_5 not found, using zeros")
        angle_downer_upper_deg = np.zeros_like(time)

    # Define colors for each state
    state_colors = {
        1: [0.0, 0.0, 0.0],      # Black (for Crawl)
        2: [1.0, 0.0, 0.0],      # Red (for Bend)
        3: [0.0, 1.0, 0.0],      # Green (for Stop)
        4: [0.0, 0.0, 1.0],      # Blue (for Hunch)
        5: [1.0, 0.5, 0.0],      # Orange
        6: [0.5, 0.0, 0.5],      # Purple
        7: [0.7, 0.7, 0.7]       # Light gray (for Small action)
    }
    
    # Define behavior names for legend
    behavior_labels = {
        1: 'Run/Crawl',
        2: 'Cast/Bend',
        3: 'Stop',
        4: 'Hunch',
        5: 'Backup',
        6: 'Roll',
        7: 'Small Actions'
    }

    # Create trajectory colors
    trajectory_colors = []
    for state in global_state.flatten():
        try:
            state_val = int(state)
            trajectory_colors.append(state_colors.get(state_val, [1, 1, 1]))
        except:
            trajectory_colors.append([1, 1, 1])

    # Create figure and axes with the requested layout
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[:, 0])  # Larva contour (takes full height of left column)
    ax2 = fig.add_subplot(gs[0, 1])  # Global trajectory (top right)
    ax3 = fig.add_subplot(gs[1, 1])  # Angle plot (bottom right)

    # Initialize left panel plots
    contour_line, = ax1.plot([], [], 'k-', linewidth=LINE_WIDTH)
    contour_fill = ax1.fill([], [], color='gray', alpha=ALPHA)[0]
    
    # Initialize spine visualization with all points
    spine_line, = ax1.plot([], [], '-', lw=LINE_WIDTH, alpha=ALPHA)
    spine_points = []
    
    # Create placeholder points for each spine point
    if x_spine.ndim > 1:
        num_spine_points = x_spine.shape[0]
    else:
        num_spine_points = 1
    
    for i in range(num_spine_points):
        point, = ax1.plot([], [], 'o', ms=SPINE_MARKER_SIZE)
        spine_points.append(point)
    
    # Initialize specific body part points
    head_point, = ax1.plot([], [], 'o', ms=MARKER_SIZE)
    tail_point, = ax1.plot([], [], 's', ms=MARKER_SIZE)
    center_point, = ax1.plot([], [], '^', ms=MARKER_SIZE)
    neck_point, = ax1.plot([], [], 'D', ms=MARKER_SIZE)

    # Initialize right top panel with colored segments
    for i in range(len(x_center)-1):
        ax2.plot(x_center[i:i+2], y_center[i:i+2], 
                color=trajectory_colors[i], 
                linewidth=LINE_WIDTH, 
                alpha=ALPHA)
    current_pos, = ax2.plot([], [], 'o', ms=MARKER_SIZE)
    
    # Prepare the angle plot (bottom right)
    # Plot the full angle data with behavior shading
    ax3.plot(time, angle_downer_upper_deg, 'k-', linewidth=1.0)
    
    # Add behavior state shading to angle plot for the full time range
    for state_val in range(1, 8):
        segments = []
        in_segment = False
        segment_start = 0
        states_flat = global_state.flatten()
        
        for i, s in enumerate(states_flat):
            if s == state_val and not in_segment:
                in_segment = True
                segment_start = i
            elif s != state_val and in_segment:
                in_segment = False
                segments.append((segment_start, i))
                
        # Handle case when segment extends to end of data
        if in_segment:
            segments.append((segment_start, len(states_flat)-1))
            
        # Shade each segment
        for seg_start, seg_end in segments:
            if seg_start < seg_end:  # Only shade non-empty segments
                ax3.axvspan(time[seg_start], time[seg_end], 
                         color=state_colors[state_val], 
                         alpha=0.3)
    
    # Initialize time marker line that will be updated
    time_marker, = ax3.plot([0, 0], [ax3.get_ylim()[0], ax3.get_ylim()[1]], 
                           'r-', linewidth=2.0)

    def update(frame):
        # Extract current frame data
        x_frame = x_contour[:, frame]
        y_frame = y_contour[:, frame]
        current_time = float(time[frame])

        # Get state and color
        try:
            state = int(global_state.flatten()[frame])
            color = state_colors.get(state, [1, 1, 1])
        except:
            color = [1, 1, 1]

        # Update contour and points
        contour_line.set_data(x_frame, y_frame)
        contour_line.set_color(color)
        contour_fill.set_xy(np.column_stack((x_frame, y_frame)))
        contour_fill.set_facecolor(color)
        
        # Extract all spine point coordinates for this frame
        if x_spine.ndim > 1:  # 2D array (multiple spine points)
            spine_x = x_spine[:, frame]
            spine_y = y_spine[:, frame]
        else:  # 1D array (single spine point)
            spine_x = np.array([x_spine[frame]])
            spine_y = np.array([y_spine[frame]])
        
        # Update the full spine line
        spine_line.set_data(spine_x, spine_y)
        spine_line.set_color(color)
        
        # Update each spine point
        for i, point in enumerate(spine_points):
            if i < len(spine_x):
                point.set_data([spine_x[i]], [spine_y[i]])
                point.set_color(color)
            else:
                point.set_data([], [])  # Hide extra points
        
        # Get specific body part coordinates
        head_x = spine_x[0] if len(spine_x) > 0 else None
        head_y = spine_y[0] if len(spine_y) > 0 else None
        tail_x = spine_x[-1] if len(spine_x) > 0 else None
        tail_y = spine_y[-1] if len(spine_y) > 0 else None
        center_x = x_center[frame]
        center_y = y_center[frame]
        neck_x = x_neck[frame]
        neck_y = y_neck[frame]
        
        # Update body part points
        head_point.set_data([head_x], [head_y])
        tail_point.set_data([tail_x], [tail_y])
        center_point.set_data([center_x], [center_y])
        neck_point.set_data([neck_x], [neck_y])
        
        # Update colors
        head_point.set_color(color)
        tail_point.set_color(color)
        center_point.set_color(color)
        neck_point.set_color(color)
        current_pos.set_data([center_x], [center_y])
        current_pos.set_color(color)
        
        # Update time marker in angle plot
        time_marker.set_data([current_time, current_time], 
                            [ax3.get_ylim()[0], ax3.get_ylim()[1]])
        
        # Update visible window in angle plot - only move the window, don't redraw
        if time[-1] - time[0] > TIME_WINDOW:
            time_start = max(time[0], current_time - TIME_WINDOW/2)
            time_end = min(time[-1], current_time + TIME_WINDOW/2)
            ax3.set_xlim(time_start, time_end)
        else:
            # If recording is shorter than TIME_WINDOW, show the full range
            ax3.set_xlim(time[0], time[-1])

        # Update zoom window for contour
        ax1.set_xlim(center_x - WINDOW_SIZE, center_x + WINDOW_SIZE)
        ax1.set_ylim(center_y - WINDOW_SIZE, center_y + WINDOW_SIZE)
        ax1.set_title(f"Time: {current_time:.2f} s")

        fig.canvas.draw_idle()
        return (contour_line, contour_fill, spine_line, *spine_points, head_point, 
                tail_point, center_point, neck_point, current_pos, time_marker)

    # Set up interactive controls
    play = Play(value=0, min=0, max=len(time)-1, step=1, interval=50, description="Play")
    slider = IntSlider(min=0, max=len(time)-1, value=0, description="Frame:",
                      continuous_update=True, layout={'width': '1000px'})
    jslink((play, 'value'), (slider, 'value'))

    def on_value_change(change):
        if change['name'] == 'value':
            update(change['new'])
    slider.observe(on_value_change)

    # Configure axes
    ax1.set_aspect("equal")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.set_title("Larva Contour")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X position")
    ax2.set_ylabel("Y position")
    ax2.set_title("Global Trajectory")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Angle downer-upper (°)\n(reversed)")
    ax3.set_title("Angle Dynamics")
    
    # Configure initial view for angle plot
    if len(time) > 0:
        time_start = time[0]
        time_end = min(time[-1], time[0] + TIME_WINDOW)
        ax3.set_xlim(time_start, time_end)

    # Create behavior state legend for bottom of contour plot
    behavior_legend_elements = [
        Patch(facecolor=state_colors[i], alpha=0.6, 
              edgecolor='none', label=behavior_labels[i])
        for i in range(1, 8) if i in state_colors
    ]
    
    # Add marker legend for contour plot (top right)
    marker_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Head'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Tail'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Center'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Neck'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=SPINE_MARKER_SIZE, label='Spine Points'),
    ]
    
    # Add marker legend to top right of contour plot
    ax1.legend(handles=marker_legend, loc='upper right', title='Body parts')
    
    # Add behavior legend to bottom of contour plot
    # We place it outside the axis area
    ax1.figure.legend(handles=behavior_legend_elements, 
                     loc='lower center', 
                     bbox_to_anchor=(0.27, 0), 
                     ncol=4,
                     fontsize=8, 
                     title='Behaviors')

    # Display controls and figure
    display(HBox([play, slider]))

    update(0)
    plt.tight_layout()
    # Make room for the bottom legend
    plt.subplots_adjust(bottom=0.2)



def save_behavioral_contour_video(trx_data, larva_id, output_path=None, fps=20):
    """
    Save the behavioral contour visualization as a video file.
    
    Args:
        trx_data: Dictionary containing larva tracking data
        larva_id: ID of the larva to visualize
        output_path: Path where to save the video (default: './larva_behavior_{larva_id}.mp4')
        fps: Frames per second for the output video (default: 20)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.animation as animation
    
    if output_path is None:
        output_path = f'./larva_behavior_{larva_id}.mp4'

    # Define visualization parameters
    WINDOW_SIZE = 2
    FIGURE_SIZE = (12, 6)
    LINE_WIDTH = 2
    MARKER_SIZE = 8
    ALPHA = 0.6

    # Get data and ensure proper shapes
    larva = trx_data[larva_id]
    x_contour = np.atleast_2d(np.array(larva['x_contour']))
    y_contour = np.atleast_2d(np.array(larva['y_contour']))
    x_center = np.array(larva['x_center']).flatten()
    y_center = np.array(larva['y_center']).flatten()
    x_spine = np.array(larva['x_spine'])
    y_spine = np.array(larva['y_spine'])
    x_neck = np.array(larva['x_neck']).flatten()
    y_neck = np.array(larva['y_neck']).flatten()
    time = np.array(larva['t']).flatten()
    global_state = np.array(larva['global_state_large_state'])

    # Define colors for each state
    state_colors = {
        1: [1, 0.7, 0.7],    # Light red (run)
        2: [0.7, 0.7, 1],    # Light blue (cast)
        3: [0.7, 1, 0.7],    # Light green (stop)
        4: [0.9, 0.7, 0.9],  # Light purple (head cast)
        5: [1, 1, 0.7],      # Light yellow (backup)
        6: [1, 0.8, 0.6],    # Light orange (collision)
        7: [1, 0.8, 0.9],    # Light pink (small actions)
    }

    # Create figure and axes
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Initialize plots
    contour_line, = ax1.plot([], [], 'k-', linewidth=LINE_WIDTH)
    contour_fill = ax1.fill([], [], color='gray', alpha=ALPHA)[0]
    spine_line, = ax1.plot([], [], '-', lw=LINE_WIDTH, alpha=ALPHA)
    head_point, = ax1.plot([], [], 'o', ms=MARKER_SIZE)
    tail_point, = ax1.plot([], [], 's', ms=MARKER_SIZE)
    center_point, = ax1.plot([], [], '^', ms=MARKER_SIZE)
    neck_point, = ax1.plot([], [], 'D', ms=MARKER_SIZE)

    # Plot full trajectory with colors
    for i in range(len(x_center)-1):
        try:
            state = int(global_state.flatten()[i])
            color = state_colors.get(state, [1, 1, 1])
        except:
            color = [1, 1, 1]
        ax2.plot(x_center[i:i+2], y_center[i:i+2], 
                color=color, linewidth=LINE_WIDTH, alpha=ALPHA)
    current_pos, = ax2.plot([], [], 'o', ms=MARKER_SIZE)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color=state_colors[1], label='Run'),
        Line2D([0], [0], color=state_colors[2], label='Cast'),
        Line2D([0], [0], color=state_colors[3], label='Stop'),
        Line2D([0], [0], color=state_colors[4], label='Head Cast'),
        Line2D([0], [0], color=state_colors[5], label='Backup'),
        Line2D([0], [0], color=state_colors[6], label='Collision'),
        Line2D([0], [0], color=state_colors[7], label='Small Actions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', 
               markersize=MARKER_SIZE, label='Head'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', 
               markersize=MARKER_SIZE, label='Tail'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', 
               markersize=MARKER_SIZE, label='Center'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='k', 
               markersize=MARKER_SIZE, label='Neck'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    def animate(frame):
        # Extract current frame data
        x_frame = x_contour[:, frame]
        y_frame = y_contour[:, frame]
        head_x, head_y = x_spine[0, frame], y_spine[0, frame]
        tail_x, tail_y = x_spine[-1, frame], y_spine[-1, frame]
        center_x, center_y = x_center[frame], y_center[frame]
        neck_x, neck_y = x_neck[frame], y_neck[frame]
        current_time = float(time[frame])

        # Get state and color
        try:
            state = int(global_state.flatten()[frame])
            color = state_colors.get(state, [1, 1, 1])
        except:
            color = [1, 1, 1]

        # Update plots
        contour_line.set_data(x_frame, y_frame)
        contour_line.set_color(color)
        contour_fill.set_xy(np.column_stack((x_frame, y_frame)))
        contour_fill.set_facecolor(color)
        
        spine_line.set_data([head_x, center_x, tail_x], [head_y, center_y, tail_y])
        head_point.set_data([head_x], [head_y])
        tail_point.set_data([tail_x], [tail_y])
        center_point.set_data([center_x], [center_y])
        neck_point.set_data([neck_x], [neck_y])
        
        # Update colors
        spine_line.set_color(color)
        head_point.set_color(color)
        tail_point.set_color(color)
        center_point.set_color(color)
        neck_point.set_color(color)
        current_pos.set_data([center_x], [center_y])
        current_pos.set_color(color)

        # Update zoom window and title
        ax1.set_xlim(center_x - WINDOW_SIZE, center_x + WINDOW_SIZE)
        ax1.set_ylim(center_y - WINDOW_SIZE, center_y + WINDOW_SIZE)
        ax1.set_title(f"Time: {current_time:.2f} s")

        return (contour_line, contour_fill, spine_line, head_point, 
                tail_point, center_point, neck_point, current_pos)

    # Configure axes
    ax1.set_aspect("equal")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X position")
    ax2.set_ylabel("Y position")
    ax2.set_title("Global Trajectory")

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time),
                                 interval=1000/fps, blit=True)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"Video saved to: {output_path}")


def analyze_run_orientations_single(trx_data):
    """
    Analyze run orientations for all larvae in a single experiment.
    
    Args:
        trx_data: Dictionary containing larva tracking data
        
    Returns:
        dict: Contains combined orientation data and statistics
    """
    import numpy as np
    
    all_orientations = []
    all_run_masks = []
    
    # Process each larva
    for larva_id, larva in trx_data.items():
        try:
            # Get position data - handle both nested and flat structures
            if 'data' in larva:
                larva = larva['data']
            
            # Extract and validate arrays
            x_center = np.asarray(larva['x_center']).flatten()
            y_center = np.asarray(larva['y_center']).flatten()
            x_spine = np.asarray(larva['x_spine'])
            y_spine = np.asarray(larva['y_spine'])
            states = np.asarray(larva['global_state_large_state']).flatten()
            
            # Get tail positions (last spine point)
            x_tail = x_spine[-1].flatten() if x_spine.ndim > 1 else x_spine.flatten()
            y_tail = y_spine[-1].flatten() if y_spine.ndim > 1 else y_spine.flatten()
            
            # Calculate tail-to-center vectors
            tail_to_center = np.column_stack([
                x_center - x_tail,
                y_center - y_tail
            ])
            
            # Calculate orientations in degrees
            orientations = np.degrees(np.arctan2(
                tail_to_center[:, 1],
                tail_to_center[:, 0]
            ))
            
            # Create run mask
            run_mask = states == 1
            
            all_orientations.extend(orientations[run_mask])
            all_run_masks.extend(run_mask)
            
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    return {
        'orientations': np.array(all_orientations),
        'n_larvae': len(trx_data)
    }

def analyze_run_orientations_all(experiments_data):
    """
    Analyze run orientations across all experiments.
    
    Args:
        experiments_data: Dict containing all experiments data
        
    Returns:
        dict: Contains combined orientation data and statistics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    # Initialize storage
    all_orientations = []
    total_larvae = 0
    
    # Handle nested data structure
    if 'data' in experiments_data:
        data_to_process = experiments_data['data']
        total_larvae = experiments_data['metadata']['total_larvae']
    else:
        data_to_process = experiments_data
        total_larvae = len(data_to_process)
    
    # Process each experiment
    if isinstance(data_to_process, dict):
        results = analyze_run_orientations_single(data_to_process)
        all_orientations.extend(results['orientations'])
    else:
        for exp_data in data_to_process.values():
            results = analyze_run_orientations_single(exp_data)
            all_orientations.extend(results['orientations'])
    
    # Convert to numpy array
    all_orientations = np.array(all_orientations)
    
    # Create histogram
    hist, bins = np.histogram(all_orientations, 
                            bins=36, 
                            range=(-180, 180),
                            density=True)
    
    # Plot distribution
    plt.figure(figsize=(4, 3))
    
    # Plot raw histogram
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, hist, 'k-', alpha=0.3, linewidth=1)
    
    # Add smoothed line
    smoothed = gaussian_filter1d(hist, sigma=1)
    plt.plot(bin_centers, smoothed, 'r-', linewidth=2)
    
    plt.xlabel('Run heading (°)')
    plt.ylabel('Relative probability\nof orientation')
    plt.xlim(-180, 180)
    plt.title(f'Running Orientation (n={total_larvae})')
    plt.tight_layout()
    
    return {
        'orientations': all_orientations,
        'histogram': hist,
        'bin_centers': bin_centers,
        'smoothed': smoothed,
        'n_larvae': total_larvae
    }




def analyze_turn_rate_by_orientation(trx_data, larva_id=None, bin_width=10):
    """
    Calculate turn rate as a function of orientation.
    
    Args:
        trx_data: Dictionary containing tracking data
        larva_id: Optional specific larva to analyze. If None, analyzes all larvae
        bin_width: Width of orientation bins in degrees
        
    Returns:
        dict: Contains turn rates and orientation bins
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt
    
    def get_orientations_and_states(larva_data):
        # Calculate orientation
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten()
        
        tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
        orientations = np.degrees(np.arctan2(tail_to_center[:, 1], tail_to_center[:, 0]))
        
        # Get casting states (state == 2)
        states = np.array(larva_data['global_state_large_state']).flatten()
        is_casting = states == 2
        
        return orientations, is_casting
    
    # Initialize storage
    all_orientations = []
    all_casting_states = []
    
    # Process data
    if larva_id is not None:
        # Single larva analysis
        larva_data = trx_data[larva_id]
        orientations, is_casting = get_orientations_and_states(larva_data)
        all_orientations.extend(orientations)
        all_casting_states.extend(is_casting)
        n_larvae = 1
        title = f'Larva {larva_id}'
    else:
        # All larvae analysis
        if 'data' in trx_data:
            data_to_process = trx_data['data']
            n_larvae = trx_data['metadata']['total_larvae']
        else:
            data_to_process = trx_data
            n_larvae = len(data_to_process)
            
        for larva_data in data_to_process.values():
            orientations, is_casting = get_orientations_and_states(larva_data)
            all_orientations.extend(orientations)
            all_casting_states.extend(is_casting)
        title = f'Turn Probability (n={n_larvae})'
    
    all_orientations = np.array(all_orientations)
    all_casting_states = np.array(all_casting_states)
    
    # Create orientation bins
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate turn rate for each bin
    turn_rates = []
    for i in range(len(bins)-1):
        mask = (all_orientations >= bins[i]) & (all_orientations < bins[i+1])
        if np.sum(mask) > 0:
            turn_rate = np.mean(all_casting_states[mask])
        else:
            turn_rate = 0
        turn_rates.append(turn_rate)
    
    turn_rates = np.array(turn_rates)
    
    # Plot results
    plt.figure(figsize=(4, 3))
    
    # Plot raw turn rates
    plt.plot(bin_centers, turn_rates, 'k-', alpha=0.3, linewidth=1)
    
    # Plot smoothed turn rates
    smoothed = gaussian_filter1d(turn_rates, sigma=1)
    plt.plot(bin_centers, smoothed, 'r-', linewidth=2)
    
    plt.xlabel('Orientation (°)')
    plt.ylabel('Turn probability')
    plt.xlim(-180, 180)
    plt.title(title)
    plt.tight_layout()
    
    return {
        'orientations': all_orientations,
        'casting_states': all_casting_states,
        'bin_centers': bin_centers,
        'turn_rates': turn_rates,
        'smoothed_rates': smoothed,
        'n_larvae': n_larvae
    }

def analyze_turn_amplitudes_by_orientation(trx_data, larva_id=None, bin_width=10):
    """
    Calculate turn amplitudes as a function of orientation.
    
    Turn amplitude is defined as the absolute change in orientation between 
    successive frames when a turn (state == 2) is detected.
    
    Args:
        trx_data: Dictionary containing tracking data.
        larva_id: Optional specific larva to analyze. If None, analyzes all larvae.
        bin_width: Width of orientation bins in degrees.
        
    Returns:
        dict: Contains turn amplitude data including:
            - base_orientations: the orientation (in degrees) at the frame before the turn.
            - turn_amplitudes: amplitude of turn (in degrees) for each event.
            - bin_centers: centers of orientation bins.
            - mean_amplitudes: mean turn amplitude for each bin.
            - smoothed_amplitudes: amplitudes smoothed with gaussian_filter1d.
            - n_larvae: number of larvae processed.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt

    def get_orientations_and_states(larva_data):
        # Calculate orientation using tail-to-center vector
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten()
        tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
        orientations = np.degrees(np.arctan2(tail_to_center[:, 1], tail_to_center[:, 0]))
        
        # Get turn states (here, state == 2 considered as turning/casting)
        states = np.array(larva_data['global_state_large_state']).flatten()
        is_turning = states == 2
        return orientations, is_turning

    def circular_diff(a, b):
        """
        Compute minimal difference between two angles a and b (in degrees).
        Result is in [-180, 180] and we take the absolute value.
        """
        diff = a - b
        diff = (diff + 180) % 360 - 180
        return np.abs(diff)
    
    # Storage for base orientation values and corresponding turn amplitudes
    all_base_orientations = []
    all_turn_amplitudes = []
    
    # Process data either for single larva or all larvae
    if larva_id is not None:
        larva_data = trx_data[larva_id]
        orientations, is_turning = get_orientations_and_states(larva_data)
        n_larvae = 1
        title = f'Larva {larva_id} - Turn Amplitudes'
        
        # Compute turn amplitude for frames where turning occurs (skip the first frame)
        for i in range(1, len(orientations)):
            if is_turning[i]:
                amp = circular_diff(orientations[i], orientations[i-1])
                all_turn_amplitudes.append(amp)
                all_base_orientations.append(orientations[i-1])
    else:
        if 'data' in trx_data:
            data_to_process = trx_data['data']
            n_larvae = trx_data['metadata']['total_larvae']
        else:
            data_to_process = trx_data
            n_larvae = len(data_to_process)
        title = f'Turn Amplitudes (n={n_larvae})'
        
        for larva in data_to_process.values():
            orientations, is_turning = get_orientations_and_states(larva)
            for i in range(1, len(orientations)):
                if is_turning[i]:
                    amp = circular_diff(orientations[i], orientations[i-1])
                    all_turn_amplitudes.append(amp)
                    all_base_orientations.append(orientations[i-1])
    
    all_base_orientations = np.array(all_base_orientations)
    all_turn_amplitudes = np.array(all_turn_amplitudes)
    
    # Create orientation bins from -180 to 180 degrees
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Bin the turn amplitudes by the base orientation
    mean_amplitudes = []
    for i in range(len(bins)-1):
        mask = (all_base_orientations >= bins[i]) & (all_base_orientations < bins[i+1])
        if np.sum(mask) > 0:
            mean_amp = np.mean(all_turn_amplitudes[mask])
        else:
            mean_amp = np.nan  # No events in bin
        mean_amplitudes.append(mean_amp)
    mean_amplitudes = np.array(mean_amplitudes)
    
    # Smooth the binned amplitudes for visualization
    # Replace NaNs with zeros for smoothing, then restore NaNs after smoothing if needed.
    smooth_input = np.nan_to_num(mean_amplitudes, nan=0)
    smoothed_amplitudes = gaussian_filter1d(smooth_input, sigma=1)
    
    # Plot results
    plt.figure(figsize=(4, 3))
    plt.plot(bin_centers, mean_amplitudes, 'ko-', alpha=0.3, linewidth=1, label='Raw')
    plt.plot(bin_centers, smoothed_amplitudes, 'r-', linewidth=2, label='Smoothed')
    plt.xlabel('Orientation (°)')
    plt.ylabel('Mean Turn Amplitude (°)')
    plt.xlim(-180, 180)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        'base_orientations': all_base_orientations,
        'turn_amplitudes': all_turn_amplitudes,
        'bin_centers': bin_centers,
        'mean_amplitudes': mean_amplitudes,
        'smoothed_amplitudes': smoothed_amplitudes,
        'n_larvae': n_larvae
    }

def analyze_lateral_turn_rates(trx_data, angle_width=15, bin_width=5):
    """
    Analyze turn rates around ±90° orientations with confidence intervals.
    
    Args:
        trx_data: Dictionary containing tracking data
        angle_width: Width of lateral quadrants around ±90°
        bin_width: Width of orientation bins in degrees
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    def get_orientations_and_states(larva_data):
        """Extract orientation and casting state data from a single larva."""
        try:
            # Handle nested data structure
            if 'data' in larva_data:
                larva_data = larva_data['data']
                
            # Get position data
            x_center = np.array(larva_data['x_center']).flatten()
            y_center = np.array(larva_data['y_center']).flatten()
            x_spine = np.array(larva_data['x_spine'])
            y_spine = np.array(larva_data['y_spine'])
            
            # Get tail positions
            x_tail = x_spine[-1].flatten() if x_spine.ndim > 1 else x_spine.flatten()
            y_tail = y_spine[-1].flatten() if y_spine.ndim > 1 else y_spine.flatten()
            
            # Calculate tail-to-center vectors
            tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
            orientations = np.degrees(np.arctan2(tail_to_center[:, 1], tail_to_center[:, 0]))
            
            # Get casting states
            states = np.array(larva_data['global_state_large_state']).flatten()
            is_casting = states == 2
            
            return orientations, is_casting
            
        except Exception as e:
            print(f"Error processing larva data: {str(e)}")
            return np.array([]), np.array([])
    
    # Define lateral regions
    left_range = (-90-angle_width, -90+angle_width)
    right_range = (90-angle_width, 90+angle_width)
    
    # Initialize storage
    all_orientations = []
    all_casting_states = []
    left_rates = []
    right_rates = []
    
    # Process all larvae
    if 'data' in trx_data:
        data_to_process = trx_data['data']
        n_larvae = trx_data['metadata']['total_larvae']
    else:
        data_to_process = trx_data
        n_larvae = len(data_to_process)
        
    for larva_id, larva_data in data_to_process.items():
        orientations, is_casting = get_orientations_and_states(larva_data)
        
        if len(orientations) > 0 and len(is_casting) > 0:
            # Add to overall distributions
            all_orientations.extend(orientations)
            all_casting_states.extend(is_casting)
            
            # Calculate rates for lateral quadrants
            left_mask = ((orientations >= left_range[0]) & 
                        (orientations < left_range[1]))
            right_mask = ((orientations >= right_range[0]) & 
                         (orientations < right_range[1]))
            
            if np.any(left_mask):
                left_rates.append(np.mean(is_casting[left_mask]))
            if np.any(right_mask):
                right_rates.append(np.mean(is_casting[right_mask]))
    
    # Convert to arrays
    all_orientations = np.array(all_orientations)
    all_casting_states = np.array(all_casting_states)
    left_rates = np.array(left_rates)
    right_rates = np.array(right_rates)
    
    # Create orientation bins
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate turn rates and confidence intervals for each bin
    turn_rates = []
    turn_rates_std = []
    turn_rates_sem = []
    n_samples = []
    
    for i in range(len(bins)-1):
        mask = (all_orientations >= bins[i]) & (all_orientations < bins[i+1])
        bin_data = all_casting_states[mask]
        if len(bin_data) > 0:
            turn_rates.append(np.mean(bin_data))
            turn_rates_std.append(np.std(bin_data))
            turn_rates_sem.append(stats.sem(bin_data))
            n_samples.append(len(bin_data))
        else:
            turn_rates.append(0)
            turn_rates_std.append(0)
            turn_rates_sem.append(0)
            n_samples.append(0)
    
    turn_rates = np.array(turn_rates)
    turn_rates_std = np.array(turn_rates_std)
    turn_rates_sem = np.array(turn_rates_sem)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Full distribution with confidence intervals
    smoothed = gaussian_filter1d(turn_rates, sigma=1)
    smoothed_sem = gaussian_filter1d(turn_rates_sem, sigma=1)
    
    # Plot raw data points with size proportional to sample size
    sizes = np.array(n_samples) / max(n_samples) * 100
    ax1.scatter(bin_centers, turn_rates, c='k', alpha=0.3, s=sizes)
    
    # Plot smoothed line with shaded confidence interval
    ax1.plot(bin_centers, smoothed, 'r-', linewidth=2, label='Mean turn rate')
    ax1.fill_between(bin_centers, 
                     smoothed - 1.96*smoothed_sem,
                     smoothed + 1.96*smoothed_sem,
                     color='r', alpha=0.2, label='95% CI')
    
    # Highlight lateral regions
    ax1.axvspan(left_range[0], left_range[1], color='blue', alpha=0.1)
    ax1.axvspan(right_range[0], right_range[1], color='blue', alpha=0.1)
    
    ax1.set_xlabel('Orientation (°)')
    ax1.set_ylabel('Turn probability')
    ax1.set_xlim(-180, 180)
    ax1.set_title('Turn Rate Distribution')
    ax1.legend()
    
    # Plot 2: Lateral quadrants comparison
    bp = ax2.boxplot([left_rates, right_rates],
                     positions=[0, 1],
                     labels=['Left\n(-90°)', 'Right\n(90°)'],
                     notch=True,
                     patch_artist=True)
    
    # Add individual points with jitter
    for i, data in enumerate([left_rates, right_rates]):
        if len(data) > 0:
            jitter = np.random.normal(0, 0.02, size=len(data))
            ax2.scatter(np.full_like(data, i+1) + jitter, data,
                       alpha=0.2, color='blue', label='Individual larvae' if i==0 else "")
    
    # Add mean ± SEM for each quadrant
    for i, data in enumerate([left_rates, right_rates]):
        if len(data) > 0:
            mean = np.mean(data)
            sem = stats.sem(data)
            ax2.errorbar(i+1, mean, yerr=sem, color='red', 
                        capsize=5, capthick=2, label='Mean ± SEM' if i==0 else "")
    
    # Add statistics
    stats_text = []
    for name, data in zip(['Left', 'Right'], [left_rates, right_rates]):
        if len(data) > 0:
            stats_text.append(f"{name}:\n"
                            f"mean = {np.mean(data):.3f}\n"
                            f"SEM = {stats.sem(data):.3f}\n"
                            f"n = {len(data)}")
    
    if len(left_rates) > 0 and len(right_rates) > 0:
        t_stat, p_val = stats.ttest_ind(left_rates, right_rates)
        stats_text.append(f"\nt-test p-value: {p_val:.3e}")
    
    ax2.text(1.2, ax2.get_ylim()[0], '\n'.join(stats_text),
             va='bottom', ha='left')
    
    ax2.set_ylabel('Turn probability')
    ax2.set_title('Lateral Turn Rates')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'left_rates': left_rates,
        'right_rates': right_rates,
        'bin_centers': bin_centers,
        'turn_rates': turn_rates,
        'turn_rates_sem': turn_rates_sem,
        'smoothed': smoothed,
        'n_samples': n_samples,
        'angle_width': angle_width,
        'n_larvae': n_larvae
    }

def analyze_cast_directions(trx_data, fps=6):
    """
    Analyze if casts are upstream or downstream relative to wind direction.
    
    Args:
        trx_data: Dictionary containing tracking data
        fps: Frames per second for time calculation
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    
    def get_cast_events(larva_data):
        """
        Get cast events and their angles from larva data.
        Returns list of angles during cast initiation.
        """
        try:
            # Get states and find cast start indices
            states = np.array(larva_data['global_state_large_state']).flatten()
            cast_starts = np.where((states[1:] == 2) & (states[:-1] != 2))[0] + 1
            
            # For each cast, calculate average orientation in first second
            cast_directions = []
            frames_per_sec = int(fps)
            
            for start in cast_starts:
                end = min(start + frames_per_sec, len(states))
                if end - start < frames_per_sec/2:  # Skip if cast is too short
                    print('too short ')
                    continue
                
                try:
                    # Get head vector relative to center
                    x_center = larva_data['x_center'][start:end]
                    y_center = larva_data['y_center'][start:end]
                    
                    # Extract spine data correctly
                    x_spine = np.array(larva_data['x_spine'])
                    y_spine = np.array(larva_data['y_spine'])
                    
                    # Get head coordinates (first spine point)
                    if x_spine.ndim > 1:  # If spine data is 2D
                        x_head = x_spine[0][start:end]
                        y_head = y_spine[0][start:end]
                    else:  # If spine data is 1D
                        x_head = x_spine[start:end]
                        y_head = y_spine[start:end]
                    
                    # Ensure all arrays have data
                    if (len(x_center) == 0 or len(y_center) == 0 or 
                        len(x_head) == 0 or len(y_head) == 0):
                        continue
                    
                    # Convert to numpy arrays if they aren't already
                    x_center = np.array(x_center).flatten()
                    y_center = np.array(y_center).flatten()
                    x_head = np.array(x_head).flatten()
                    y_head = np.array(y_head).flatten()
                    
                    # Calculate head vector angle
                    head_vectors = np.column_stack([x_head - x_center, y_head - y_center])
                    angles = np.degrees(np.arctan2(head_vectors[:, 1], -head_vectors[:, 0]))
                    mean_angle = np.mean(angles)
                    print(f'mean angle {mean_angle}')
                    
                    cast_directions.append(mean_angle)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error processing cast at frame {start}: {str(e)}")
                    continue
                    
            return cast_directions
            
        except Exception as e:
            print(f"Error in get_cast_events: {str(e)}")
            return []
    
    # Process all larvae
    upstream_casts = []
    downstream_casts = []
    
    if 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data
        
    for larva_id, larva_data in data_to_process.items():
        try:
            if 'data' in larva_data:
                larva_data = larva_data['data']
                
            cast_angles = get_cast_events(larva_data)
            # Classify casts
            for angle in cast_angles:
                print(angle)
                # Normalize angle to [-180, 180]
                angle = ((angle + 180) % 360) - 180
                if -90 <= angle <= 90:
                    upstream_casts.append(angle)
                else:
                    downstream_casts.append(angle)
                    
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    # Check if we have any data to plot
    if len(upstream_casts) == 0 and len(downstream_casts) == 0:
        print("No valid cast events found in the data")
        return None
        
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create box plots instead of violin plots (more robust with small datasets)
    positions = [1, 2]
    data = [upstream_casts, downstream_casts]
    labels = ['Upstream\ncasts', 'Downstream\ncasts']
    
    # Create box plots
    bp = plt.boxplot(data, positions=positions, 
                    notch=True, 
                    patch_artist=True,
                    showfliers=False)
    
    # Customize colors
    colors = ['#4DBBD5', '#00A087']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    
    # Add individual points with jitter
    for i, d in enumerate(data):
        if len(d) > 0:
            pos = positions[i]
            jitter = np.random.normal(0, 0.02, size=len(d))
            plt.scatter(np.full_like(d, pos) + jitter, d,
                       alpha=0.2, color=colors[i], s=20)
    
    # Add statistics
    stats_text = []
    for name, d in zip(['Upstream', 'Downstream'], data):
        if len(d) > 0:
            stats_text.append(f"{name}:\n"
                            f"mean = {np.mean(d):.1f}°\n"
                            f"std = {np.std(d):.1f}°\n"
                            f"n = {len(d)}")
    
    # Perform statistical test
    if len(upstream_casts) > 0 and len(downstream_casts) > 0:
        t_stat, p_val = stats.ttest_ind(upstream_casts, downstream_casts)
        stats_text.append(f"\nt-test p-value: {p_val:.3e}")
    
    plt.text(2.3, plt.ylim()[0], '\n'.join(stats_text),
             va='bottom', ha='left')
    
    # Customize plot
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xticks(positions, labels)
    plt.ylabel('Cast angle (degrees)')
    plt.title('Distribution of Cast Directions\nRelative to Wind (-x axis)')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'upstream_angles': np.array(upstream_casts),
        'downstream_angles': np.array(downstream_casts),
        'upstream_mean': np.mean(upstream_casts) if len(upstream_casts) > 0 else np.nan,
        'downstream_mean': np.mean(downstream_casts) if len(downstream_casts) > 0 else np.nan,
        'upstream_std': np.std(upstream_casts) if len(upstream_casts) > 0 else np.nan,
        'downstream_std': np.std(downstream_casts) if len(downstream_casts) > 0 else np.nan,
        'n_upstream': len(upstream_casts),
        'n_downstream': len(downstream_casts)
    }


def analyze_perpendicular_cast_directions(trx_data, angle_width=15):
    """
    Analyze the probability of upstream vs downstream casts when larvae are perpendicular to flow.
    
    Args:
        trx_data (dict): Tracking data dictionary containing larvae data
        angle_width (int): Width of perpendicular orientation sector in degrees
    
    Returns:
        dict: Cast direction metrics including probabilities and statistical analysis
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns
    import pandas as pd
    
    # Determine which data structure we're working with
    if 'data' in trx_data:
        data_to_process = trx_data['data']
        n_larvae = trx_data['metadata']['total_larvae']
    else:
        data_to_process = trx_data
        n_larvae = len(data_to_process)
    
    # Store raw counts for analysis
    total_counts = {'upstream': 0, 'downstream': 0}
    
    # Store per-larva probabilities for violin plot
    larva_probabilities = {
        'upstream': [],
        'downstream': []
    }
    
    # Define perpendicular angle ranges (both left and right sides)
    left_perp_range = (-90 - angle_width, -90 + angle_width)
    right_perp_range = (90 - angle_width, 90 + angle_width)
    
    def is_perpendicular(angle):
        """Check if angle is in perpendicular range"""
        return ((left_perp_range[0] <= angle <= left_perp_range[1]) or 
                (right_perp_range[0] <= angle <= right_perp_range[1]))
    
    def determine_cast_direction(init_angle, cast_angle):
        """
        Determine if cast is upstream or downstream when larva is perpendicular.
        For left-side perpendicular (-90°), upstream = negative angle difference
        For right-side perpendicular (90°), upstream = positive angle difference
        """
        angle_diff = (cast_angle - init_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
            
        # If larva is around -90 (left perpendicular)
        if left_perp_range[0] <= init_angle <= left_perp_range[1]:
            return 'upstream' if angle_diff < 0 else 'downstream'
        # If larva is around 90 (right perpendicular)
        else:
            return 'upstream' if angle_diff > 0 else 'downstream'
    
    # Process each larva
    larvae_processed = 0
    for larva_id, larva_data in data_to_process.items():
        try:
            # Extract nested data if needed
            if 'data' in larva_data:
                larva_data = larva_data['data']
            
            # Extract behavioral states
            cast = np.array(larva_data.get('cast', larva_data.get('global_state_large_state', [])))
            if len(cast) == 0:
                continue
                
            # Find cast bout starts
            cast_starts = np.where((cast[1:] == 2) & (cast[:-1] != 2))[0] + 1
            if len(cast_starts) == 0:
                # Alternative: just use any frame where cast == 2
                cast_starts = np.where(cast == 2)[0]
                if len(cast_starts) == 0:
                    continue
            
            # Extract coordinates
            x_spine = np.array(larva_data['x_spine'])
            y_spine = np.array(larva_data['y_spine'])
            x_center = np.array(larva_data['x_center']).flatten()
            y_center = np.array(larva_data['y_center']).flatten()
            
            # Handle different spine data shapes
            if x_spine.ndim == 1:  # 1D array
                x_tail = x_spine
                y_tail = y_spine
                x_head = x_spine
                y_head = y_spine
            else:  # 2D array with shape (spine_points, frames)
                x_tail = x_spine[-1, :]
                y_tail = y_spine[-1, :]
                x_head = x_spine[0, :]
                y_head = y_spine[0, :]
            
            # Calculate vectors
            tail_to_center = np.column_stack([
                x_center - x_tail,
                y_center - y_tail
            ])
            
            center_to_head = np.column_stack([
                x_head - x_center,
                y_head - y_center
            ])
            
            # Calculate angles
            orientation_angles = np.degrees(np.arctan2(tail_to_center[:, 1], -tail_to_center[:, 0]))
            cast_angles = np.degrees(np.arctan2(center_to_head[:, 1], -center_to_head[:, 0]))
            
            # Initialize counts for this larva
            larva_counts = {'upstream': 0, 'downstream': 0}
            
            # Analyze each cast bout
            for start in cast_starts:
                try:
                    if start >= len(orientation_angles):
                        continue
                        
                    # Get initial orientation (at start of cast)
                    init_orientation = orientation_angles[start]
                    
                    # Only analyze casts when orientation is perpendicular
                    if not is_perpendicular(init_orientation):
                        continue
                        
                    # Get maximum cast angle
                    end = min(start + 6, len(cast_angles))  # Look at first 6 frames of cast
                    if end <= start or end >= len(cast_angles):
                        continue
                        
                    cast_sequence = cast_angles[start:end]
                    if len(cast_sequence) < 3:  # Need at least 3 frames
                        continue
                    
                    # Find frame with maximum deviation
                    angle_diffs = np.abs(cast_sequence - init_orientation)
                    max_deviation_idx = np.argmax(angle_diffs)
                    max_cast_angle = cast_sequence[max_deviation_idx]
                    
                    # Determine if cast is upstream or downstream
                    cast_direction = determine_cast_direction(init_orientation, max_cast_angle)
                    
                    # Update counts
                    total_counts[cast_direction] += 1
                    larva_counts[cast_direction] += 1
                        
                except (IndexError, ValueError):
                    continue
            
            # Calculate per-larva probabilities if enough casts
            larva_total = sum(larva_counts.values())
            if larva_total >= 3:  # Only include larvae with at least 3 perpendicular casts
                larva_probabilities['upstream'].append(larva_counts['upstream'] / larva_total)
                larva_probabilities['downstream'].append(larva_counts['downstream'] / larva_total)
                larvae_processed += 1
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    # Calculate overall probabilities
    total_casts = sum(total_counts.values())
    if total_casts > 0:
        probabilities = {
            'upstream': total_counts['upstream'] / total_casts,
            'downstream': total_counts['downstream'] / total_casts
        }
    else:
        probabilities = {'upstream': 0, 'downstream': 0}
    
    # Create dataframe for plotting
    plot_data = []
    for i, prob in enumerate(larva_probabilities['upstream']):
        plot_data.append({'Direction': 'Upstream', 'Probability': prob})
    for i, prob in enumerate(larva_probabilities['downstream']):
        plot_data.append({'Direction': 'Downstream', 'Probability': prob})
    df = pd.DataFrame(plot_data)
    
    # Statistical tests
    # 1. Chi-square on raw counts
    observed = np.array([total_counts['upstream'], total_counts['downstream']])
    expected = np.sum(observed) / 2  # Expected equal distribution
    chi2, p_chi2 = stats.chisquare(observed)
    
    # 2. Paired t-test on larva probabilities (upstream vs 1-upstream)
    up_probs = np.array(larva_probabilities['upstream'])
    down_probs = np.array(larva_probabilities['downstream'])
    
    # These should sum to 1 for each larva, but let's use a paired test to be sure
    if len(up_probs) > 1:
        t_stat, p_val = stats.ttest_rel(up_probs, down_probs)
    else:
        t_stat, p_val = None, None
    
    # Create violin plot
    plt.figure(figsize=(7, 5))
    
    # Create violin plot with individual data points
    sns.violinplot(x='Direction', y='Probability', data=df, inner=None, 
                  palette={'Upstream': 'green', 'Downstream': 'orange'})
    
    # Add individual data points
    sns.stripplot(x='Direction', y='Probability', data=df, color='black', 
                 size=5, alpha=0.5, jitter=True)
    
    # Add significance bar if p-value is significant
    if p_val is not None and p_val < 0.05:
        # Get y positions
        y_max = max(max(up_probs), max(down_probs)) + 0.05
        
        # Plot the line
        x1, x2 = 0, 1
        plt.plot([x1, x1, x2, x2], [y_max, y_max + 0.03, y_max + 0.03, y_max], lw=1.5, c='black')
        
        # Add stars based on significance level
        stars = '*' * sum([p_val < p for p in [0.05, 0.01, 0.001]])
        plt.text((x1 + x2) * 0.5, y_max + 0.04, stars, ha='center', va='bottom', color='black', fontsize=14)
        
        # Add p-value
        plt.text((x1 + x2) * 0.5, y_max + 0.05, f'p = {p_val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add reference line for 0.5 probability (chance level)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    
    # Add raw counts as text on plot
    plt.text(0, 0.05, f'n = {total_counts["upstream"]}', ha='center', fontsize=10)
    plt.text(1, 0.05, f'n = {total_counts["downstream"]}', ha='center', fontsize=10)
    
    # Format plot
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 1.1)
    plt.title(f'Probability of Cast Direction When Perpendicular to Flow\n(±{angle_width}° around ±90°, n={larvae_processed} larvae)', 
              fontsize=14)
    
    # Add aggregate probabilities as text
    plt.text(0, probabilities['upstream'] + 0.03, f"{probabilities['upstream']*100:.1f}%", 
            ha='center', va='bottom', fontsize=12, color='green', fontweight='bold')
    plt.text(1, probabilities['downstream'] + 0.03, f"{probabilities['downstream']*100:.1f}%", 
            ha='center', va='bottom', fontsize=12, color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'total_counts': total_counts,
        'probabilities': probabilities,
        'larva_probabilities': larva_probabilities,
        'chi2_result': (chi2, p_chi2),
        'ttest_result': (t_stat, p_val) if t_stat is not None else None,
        'larvae_processed': larvae_processed,
        'total_larvae': n_larvae,
        'angle_width': angle_width
    }


def analyze_cast_head_dynamics(trx_data, larva_id=None, max_events=10):
    """
    Analyze head angle dynamics during casting events.
    
    Args:
        trx_data (dict): Tracking data dictionary
        larva_id (str, optional): ID of specific larva to analyze, if None, selects a random larva
        max_events (int): Maximum number of cast events to plot
    
    Returns:
        dict: Statistics about cast events and head angle dynamics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import random
    
    # Helper function to extract cast events from a single larva
    def extract_cast_events(larva_data):
        # Get behavioral states
        states = np.array(larva_data['global_state_large_state']).flatten()
        
        # Find cast bout starts and ends
        cast_starts = np.where((states[1:] == 2) & (states[:-1] != 2))[0] + 1
        cast_ends = []
        
        for start in cast_starts:
            end_idx = np.where(states[start:] != 2)[0]
            if len(end_idx) > 0:
                cast_ends.append(start + end_idx[0])
            else:
                cast_ends.append(len(states) - 1)  # End of recording
        
        # Ensure we have the same number of starts and ends
        n_events = min(len(cast_starts), len(cast_ends))
        cast_starts = cast_starts[:n_events]
        cast_ends = cast_ends[:n_events]
        
        # Ensure each cast has a valid duration
        valid_casts = []
        for i in range(n_events):
            if cast_ends[i] - cast_starts[i] >= 3:  # At least 3 frames
                valid_casts.append((cast_starts[i], cast_ends[i]))
        
        return valid_casts
    
    # Helper function to calculate angle between vectors
    def calculate_angle_between_vectors(v1, v2):
        """Calculate angle between two vectors in degrees"""
        dot_product = np.sum(v1 * v2, axis=1)
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        
        # Avoid division by zero
        valid_mask = (norm_v1 > 0) & (norm_v2 > 0)
        if not np.any(valid_mask):
            return np.zeros(len(v1))
            
        cos_angle = np.zeros(len(v1))
        cos_angle[valid_mask] = dot_product[valid_mask] / (norm_v1[valid_mask] * norm_v2[valid_mask])
        
        # Clip to handle floating point errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        
        return angles
    
    # Determine which larva to analyze
    if 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data
    
    if larva_id is None:
        # Select a random larva with good data
        larvae_with_casts = []
        for lid, larva_data in data_to_process.items():
            try:
                # Extract larva data if nested
                if 'data' in larva_data:
                    larva_data = larva_data['data']
                
                # Check if larva has spine data and state data
                if ('x_spine' in larva_data and 'y_spine' in larva_data and
                    'x_center' in larva_data and 'y_center' in larva_data and
                    'global_state_large_state' in larva_data):
                    events = extract_cast_events(larva_data)
                    if len(events) > 0:
                        larvae_with_casts.append(lid)
            except Exception as e:
                print(f"Error checking larva {lid}: {str(e)}")
                continue
        
        if not larvae_with_casts:
            raise ValueError("No larvae with valid cast and angle data found")
        
        larva_id = random.choice(larvae_with_casts)
        print(f"Selected random larva: {larva_id}")
    
    # Get data for the selected larva
    larva_data = data_to_process[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract data arrays
    try:
        x_spine = np.array(larva_data['x_spine'])
        y_spine = np.array(larva_data['y_spine'])
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        time = np.array(larva_data['t']).flatten()
        states = np.array(larva_data['global_state_large_state']).flatten()
        angle_downer_upper = np.array(larva_data['angle_downer_upper_smooth_5']).flatten()
    except KeyError as e:
        raise KeyError(f"Missing required data field: {str(e)}")
    
    # Convert angle_downer_upper to degrees and reverse direction
    # (assuming it's in radians, multiply by -1 to reverse)
    angle_downer_upper_deg = -1 * np.degrees(angle_downer_upper)
    
    # Handle different spine data shapes
    if x_spine.ndim == 1:  # 1D array
        x_tail = np.copy(x_spine)
        y_tail = np.copy(y_spine)
        x_head = np.copy(x_spine)
        y_head = np.copy(y_spine)
    else:  # 2D array with shape (spine_points, frames)
        x_tail = x_spine[-1, :] if len(x_spine) > 1 else x_spine.flatten()
        y_tail = y_spine[-1, :] if len(y_spine) > 1 else y_spine.flatten()
        x_head = x_spine[0, :] if len(x_spine) > 1 else x_spine.flatten()
        y_head = y_spine[0, :] if len(y_spine) > 1 else y_spine.flatten()
    
    # Calculate vectors
    tail_to_center = np.column_stack([
        x_center - x_tail,
        y_center - y_tail
    ])
    
    center_to_head = np.column_stack([
        x_head - x_center,
        y_head - y_center
    ])
    
    # Calculate angle between tail-to-center and center-to-head vectors
    bend_angle = calculate_angle_between_vectors(tail_to_center, center_to_head)
    
    # Extract cast events
    cast_events = extract_cast_events(larva_data)
    
    # Limit number of events to plot
    if len(cast_events) > max_events:
        # Select a diverse sample of cast events (short and long)
        durations = [end - start for start, end in cast_events]
        # Sort events by duration
        sorted_indices = np.argsort(durations)
        # Select events spread across duration range
        indices_to_use = sorted_indices[::len(sorted_indices)//max_events][:max_events]
        plot_events = [cast_events[i] for i in indices_to_use]
    else:
        plot_events = cast_events
    
    # Create figure
    n_events = len(plot_events)
    if n_events == 0:
        print("No cast events found for this larva")
        return None
    
    fig = plt.figure(figsize=(12, 2 * n_events))
    gs = GridSpec(n_events, 2, figure=fig)
    
    # Plot each cast event
    for i, (start, end) in enumerate(plot_events):
        # Ensure we have a complete window
        window_start = max(0, start - 5)  # 5 frames before cast
        window_end = min(len(time), end + 5)  # 5 frames after cast
        
        # Extract data for this window
        t_window = time[window_start:window_end] - time[window_start]
        angle_du_window = angle_downer_upper_deg[window_start:window_end]
        bend_angle_window = bend_angle[window_start:window_end]
        states_window = states[window_start:window_end]
        
        # Create cast mask for highlighting
        cast_mask = states_window == 2
        
        # Plot angle_downer_upper (converted to degrees and reversed)
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(t_window, angle_du_window, 'b-')
        
        # Highlight cast period
        cast_times = t_window[cast_mask]
        if len(cast_times) > 0:
            ax1.axvspan(cast_times[0], cast_times[-1], alpha=0.2, color='red')
        
        ax1.set_ylabel('Angle downer-upper (°)\n(reversed)')
        if i == n_events - 1:
            ax1.set_xlabel('Time (s)')
        else:
            ax1.set_xticklabels([])
        
        # Plot bend angle between tail-to-center and center-to-head
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(t_window, bend_angle_window, 'g-')
        
        # Highlight cast period
        if len(cast_times) > 0:
            ax2.axvspan(cast_times[0], cast_times[-1], alpha=0.2, color='red')
        
        ax2.set_ylabel('Bend angle (°)\n(tail-center-head)')
        if i == n_events - 1:
            ax2.set_xlabel('Time (s)')
        else:
            ax2.set_xticklabels([])
        
        # Add event details as title
        duration_sec = (time[end] - time[start])
        ax1.set_title(f'Event {i+1}: Duration = {duration_sec:.2f}s')
    
    plt.tight_layout()
    plt.suptitle(f"Cast Angle Dynamics - Larva {larva_id}", y=1.02, fontsize=14)
    plt.show()
    
    # Compute statistics
    all_durations = [(time[end] - time[start]) for start, end in cast_events]
    max_angle_du = []
    max_bend_angle = []
    angle_amplitude_du = []  # Range of angles during cast
    bend_amplitude = []
    
    for start, end in cast_events:
        du_during_cast = angle_downer_upper_deg[start:end]
        bend_during_cast = bend_angle[start:end]
        
        if len(du_during_cast) > 0 and len(bend_during_cast) > 0:
            max_angle_du.append(np.max(np.abs(du_during_cast)))
            max_bend_angle.append(np.max(bend_during_cast))
            
            # Calculate range/amplitude of angles during cast
            angle_amplitude_du.append(np.max(du_during_cast) - np.min(du_during_cast))
            bend_amplitude.append(np.max(bend_during_cast) - np.min(bend_during_cast))
    
    # Return statistics
    return {
        'larva_id': larva_id,
        'n_events': len(cast_events),
        'event_durations': all_durations,
        'mean_duration': np.mean(all_durations) if all_durations else None,
        'max_angle_du': max_angle_du,
        'max_bend_angle': max_bend_angle,
        'mean_max_angle_du': np.mean(max_angle_du) if max_angle_du else None,
        'mean_max_bend_angle': np.mean(max_bend_angle) if max_bend_angle else None,
        'mean_angle_amplitude_du': np.mean(angle_amplitude_du) if angle_amplitude_du else None,
        'mean_bend_amplitude': np.mean(bend_amplitude) if bend_amplitude else None
    }


def plot_larva_angle_dynamics(trx_data, larva_id=None):
    """
    Plot complete angle and velocity dynamics over time for a larva with behavior states highlighted.
    
    Args:
        trx_data (dict): Tracking data dictionary
        larva_id (str, optional): ID of specific larva to analyze, if None, selects a random larva
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from matplotlib.patches import Patch
    
    # Define behavior color scheme
    behavior_colors = {
        1: [0.0, 0.0, 0.0],      # Black (for Run/Crawl)
        2: [1.0, 0.0, 0.0],      # Red (for Cast/Bend)
        3: [0.0, 1.0, 0.0],      # Green (for Stop)
        4: [0.0, 0.0, 1.0],      # Blue (for Hunch)
        5: [1.0, 0.5, 0.0],      # Orange (for Backup)
        6: [0.5, 0.0, 0.5],      # Purple (for Roll)
        7: [0.7, 0.7, 0.7]       # Light gray (for Small Actions)
    }
    
    # Behavior labels for legend
    behavior_labels = {
        1: 'Run/Crawl',
        2: 'Cast/Bend',
        3: 'Stop',
        4: 'Hunch',
        5: 'Backup',
        6: 'Roll',
        7: 'Small Actions'
    }
    
    # Determine which data structure we're working with
    if 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data
    
    # Select a larva if not specified
    if larva_id is None:
        valid_larvae = []
        for lid, larva_data in data_to_process.items():
            try:
                # Extract larva data if nested
                if 'data' in larva_data:
                    larva_data = larva_data['data']
                
                # Check if larva has required data
                if ('head_velocity_norm_smooth_5' in larva_data and 
                    'tail_velocity_norm_smooth_5' in larva_data and
                    'angle_downer_upper_smooth_5' in larva_data and
                    'angle_upper_lower_smooth_5' in larva_data and
                    'global_state_large_state' in larva_data):
                    valid_larvae.append(lid)
            except:
                continue
        
        if not valid_larvae:
            raise ValueError("No larvae with valid data found")
        
        # Select exactly one random larva
        larva_id = random.choice(valid_larvae)
        print(f"Selected random larva: {larva_id}")
    
    # Get data for the selected larva
    larva_data = data_to_process[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract data arrays
    try:
        head_velocity = np.array(larva_data['head_velocity_norm_smooth_5']).flatten()
        tail_velocity = np.array(larva_data['tail_velocity_norm_smooth_5']).flatten()
        angle_downer_upper = np.array(larva_data['angle_downer_upper_smooth_5']).flatten()
        angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
        time = np.array(larva_data['t']).flatten()
        states = np.array(larva_data['global_state_large_state']).flatten()
        
        # Check for sufficient data
        if len(time) < 10:  # Require at least 10 frames
            raise ValueError(f"Insufficient data for larva {larva_id}")
            
    except KeyError as e:
        raise KeyError(f"Missing required data field for larva {larva_id}: {str(e)}")
    
    # Convert angles to degrees
    angle_downer_upper_deg = np.degrees(angle_downer_upper)
    angle_upper_lower_deg = np.degrees(angle_upper_lower)
    
    # Ensure all arrays have the same length
    min_length = min(len(time), len(head_velocity), len(tail_velocity), 
                    len(angle_downer_upper_deg), len(angle_upper_lower_deg), len(states))
    
    time = time[:min_length]
    head_velocity = head_velocity[:min_length]
    tail_velocity = tail_velocity[:min_length]
    angle_downer_upper_deg = angle_downer_upper_deg[:min_length]
    angle_upper_lower_deg = angle_upper_lower_deg[:min_length]
    states = states[:min_length]
    
    # Create figure with 4 stacked plots sharing x-axis
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 6), sharex=True)
    
    # Plot 1: Head velocity
    ax1.plot(time, head_velocity, 'k-', linewidth=1.5)
    ax1.set_ylabel('Head velocity\n(smooth 5)', fontsize=12)
    ax1.set_title(f'Motion Dynamics for Larva {larva_id}', fontsize=14)
    
    # Plot 2: Tail velocity
    ax2.plot(time, tail_velocity, 'k-', linewidth=1.5)
    ax2.set_ylabel('Tail velocity\n(smooth 5)', fontsize=12)
    
    # Plot 3: Angle downer-upper
    ax3.plot(time, angle_downer_upper_deg, 'k-', linewidth=1.5)
    ax3.set_ylabel('Angle downer-upper (°)', fontsize=12)
    
    # Plot 4: Angle upper-lower
    ax4.plot(time, angle_upper_lower_deg, 'k-', linewidth=1.5)
    ax4.set_xlabel('Time (seconds)', fontsize=12)
    ax4.set_ylabel('Angle upper-lower (°)', fontsize=12)
    
    # Add behavioral state highlighting to all plots
    for i in range(1, 8):  # For each behavior state (1-7)
        # Find continuous segments of this behavior
        behavior_segments = []
        in_segment = False
        segment_start = 0
        
        for j in range(len(states)):
            if states[j] == i and not in_segment:
                in_segment = True
                segment_start = j
            elif states[j] != i and in_segment:
                in_segment = False
                behavior_segments.append((segment_start, j))
        
        # If we're still in a segment at the end
        if in_segment:
            behavior_segments.append((segment_start, len(states)-1))
        
        # Highlight each segment on all plots
        for start, end in behavior_segments:
            if end > start:  # Check for valid segment
                color = behavior_colors[i]
                alpha = 0.3
                
                # Highlight on all four plots
                ax1.axvspan(time[start], time[end], color=color, alpha=alpha)
                ax2.axvspan(time[start], time[end], color=color, alpha=alpha)
                ax3.axvspan(time[start], time[end], color=color, alpha=alpha)
                ax4.axvspan(time[start], time[end], color=color, alpha=alpha)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=behavior_colors[i], alpha=0.3, 
              edgecolor='none', label=behavior_labels[i])
        for i in range(1, 8) if i in behavior_colors
    ]
    ax1.legend(handles=legend_elements, loc='upper right', 
              title='Behaviors', ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Plotted motion dynamics for larva {larva_id} over {time[-1] - time[0]:.1f} seconds")
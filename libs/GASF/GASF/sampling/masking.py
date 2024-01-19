import torch

import numpy as np



def masking(
    glitch_info: dict,
    segment_duration: float,
    segment_start_time: float=0,
    shift_range: float = 3, 
    pad_width: float = 1.5, # Make this default to half of the kernel width
    sample_rate: int=4096, 
    merge_edges: bool=True
)->dict:
    
    """Provide a buffer mask the covers the glitch at the center of the kernel.
    

    Args:
        glitch_info (dict): Glitch trigger times by each detector.
        segment_duration (float): Duration of the background.
        segment_start_time (float): Start time of the background. Defaults to 0.
        kernel_width (float, optional): The time width to cover a glitch signal.
        The unit is second. Defaults to 3.
        pad_width (float, optional)): 
        sample_rate (int, optional): The sampling rate of the background. Defaults to 4096.
        merge_edges (bool, optional): If true it will autometically conbine glitch masks 
        if the two kernels overlap.

    Returns:
        dict: A mask that labes the idxs that covers all glitch and edges 
        by the kernel start idx and end idx for each detectors. 
    """
    
    mask_kernel = {}
    if pad_width < shift_range/2:
        raise AttributeError(f"pad_width {pad_width} is shorter than half of the kernel_width {shift_range/2}")
    
    half_window = int(shift_range*sample_rate/2)
    seg_idx_count = segment_duration*sample_rate
    

    
    for ifo, glitch_time in glitch_info.items():
        
        # Initialing the first digits in the active segments aline to t0 = 0_sec
        glitch_time -= segment_start_time
        
        # Pop out glitch that lives in the edges
        ### This popping may need another argument passing.
        glitch_time = glitch_time[glitch_time > pad_width]
        glitch_time = glitch_time[glitch_time < segment_duration - pad_width]
        
        glitch_counts = len(glitch_time)
        mask_kernel[ifo] = np.zeros((glitch_counts+2, 2)).astype("int")
        
        # Provde the pad out edges mask
        mask_kernel[ifo][0, :] = np.array([0, pad_width*sample_rate])
        mask_kernel[ifo][-1, :] = np.array([seg_idx_count-pad_width*sample_rate, seg_idx_count])
        
        # Collecting the mask by idx
        glitch_idx = (glitch_time * 4096).astype("int")
        
        mask_kernel[ifo][1:-1, 0] = (glitch_idx - half_window)
        mask_kernel[ifo][1:-1, 1] = (glitch_idx + half_window)
        
    
    if merge_edges:
        
        for ifo, mask in mask_kernel.items():
            
            mask_counts = mask.shape[0]
            for i in range(mask_counts -1 ):
                
                if mask[i,1] > mask[i+1,0]:
                    mask[i,1] = mask[i+1,0]
                    
                    
    return mask_kernel


def filtering_idxs(
    mask_dict: dict,
    *n_idxs: int,
    full: bool=False,
):
    """Find segments that 

    Takes in the labeles 
    Args:
        mask_dict (dict): _description_
        segment_dur (float): _description_
        kernel_width (int, optional): _description_. Defaults to 2.
        sample_rate (int, optional): _description_. Defaults to 4096.
        shuffle (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    idx_dict = {}
    for ifo, mask in mask_dict.items():
    
        glitch_counts = len(mask)

        sampling_idx = []

        for i in range(glitch_counts-1):
            
            # Collecting usefull segments by its idx
            sampling_idx.append(torch.arange(mask[i,1], mask[i+1,0]))
            
        collected_idx = torch.cat(sampling_idx)
        
        
        if full:
            
            idx_dict[ifo] = collected_idx
        
            
        sampling_idx = torch.randint(0, len(collected_idx), n_idxs)

        idx_dict[ifo] = collected_idx[sampling_idx]
    
    return idx_dict


def strain_sampling(
    strain,
    mask: dict,
    sample_counts,
    sample_rate = 4096,
    kernel_width = 2,
):

    half_kernel_width_idx = int(kernel_width * sample_rate / 2)
    
    sampled_strain = torch.zeros([sample_counts, len(mask), sample_rate*kernel_width])

    # Cosider remove this part out of the function
    sampling_idx = filtering_idxs(
        mask, 
        sample_counts,
    )

    for _ , idxs in sampling_idx.items():
        for i, idx in enumerate(idxs):

            sampled_strain[i,:,:] = strain[:, idx-half_kernel_width_idx:idx+half_kernel_width_idx]
        
    return sampled_strain


def glitch_sampler(
    gltich_info,
    strain,
    segment_duration,
    segment_start_time,
    ifos,
    sample_counts,
    sample_rate = 4096,
    shift_range = 0.9,
    kernel_width = 3,
):
    
    half_kernel_width_idx = int(kernel_width * sample_rate / 2)
    
    sampled_strain = torch.zeros([sample_counts, len(ifos), sample_rate*kernel_width])
    
    mask_dict = masking(
        gltich_info,
        segment_duration=segment_duration,
        segment_start_time=segment_start_time,
        shift_range=shift_range,
        pad_width=kernel_width/2,
        sample_rate=sample_rate, 
        merge_edges = False
    )
    
    for i, ifo in enumerate(ifos):
        
        # Remove the padding mask
        mask_dict[ifo] = mask_dict[ifo][1:-1]
        
        glitch_count = len(mask_dict[ifo])
        selected_glitch = np.random.randint(0, glitch_count, (sample_counts,))
        sample_center = np.random.randint(
            mask_dict[ifo][selected_glitch][:, 0], 
            mask_dict[ifo][selected_glitch][:, 1], 
            size=(sample_counts)
        )
        
        for j in range(sample_counts):
            
            start_idx = sample_center[j] - half_kernel_width_idx 
            end_idx = sample_center[j] + half_kernel_width_idx
            
            sampled_strain[j, i, :] = strain[i, start_idx: end_idx]

    return sampled_strain
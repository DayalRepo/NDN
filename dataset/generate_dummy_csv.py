import pandas as pd
import numpy as np
import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(DATA_DIR, exist_ok=True)

def generate_dummy_data(num_samples=2000):
    """
    Generates dummy ndnSIM traffic logs with derived features.
    Simulates Normal traffic, IFA, Slow IFA, and Cache Pollution.
    """
    data = []
    
    # Define attack modes
    modes = ['Normal', 'IFA', 'Slow_IFA', 'Cache_Pollution', 'Distributed_IFA', 'Pulsing_IFA']
    
    for i in range(num_samples):
        mode = random.choices(modes, weights=[0.4, 0.12, 0.12, 0.12, 0.12, 0.12])[0]
        
        node = f"Node_{random.randint(0, 9)}"
        time_sec = float(i)
        
        if mode == 'Normal': # Normal Traffic
            in_interests = random.randint(50, 200)
            in_data = int(in_interests * random.uniform(0.9, 1.0))
            in_nacks = random.randint(0, 5)
            in_satisfied = in_data
            in_timedout = in_interests - in_satisfied - in_nacks
            out_interests = in_interests
            out_data = in_data
            
        elif mode == 'IFA': # Fast Interest Flooding Attack
            in_interests = random.randint(800, 2000)
            in_data = random.randint(10, 50) # Very few datas returned
            in_nacks = random.randint(10, 100)
            in_satisfied = in_data
            in_timedout = in_interests - in_satisfied - in_nacks
            out_interests = in_interests
            out_data = in_data
            
        elif mode == 'Slow_IFA': # Stealthy Attack
            in_interests = random.randint(250, 400) # Slightly above normal
            in_data = random.randint(20, 100) # Half returned
            in_nacks = random.randint(5, 20)
            in_satisfied = in_data
            in_timedout = in_interests - in_satisfied - in_nacks
            out_interests = in_interests
            out_data = in_data
            
        elif mode == 'Cache_Pollution':
            # High data return, but cache efficiency drops (simulated here by high rates)
            in_interests = random.randint(300, 600)
            in_data = int(in_interests * random.uniform(0.7, 0.85)) 
            in_nacks = random.randint(10, 30)
            in_satisfied = in_data
            in_timedout = in_interests - in_satisfied - in_nacks
            out_interests = in_interests
            out_data = in_data

        elif mode == 'Distributed_IFA':
            # Multiple attackers, moderate high interest, low data
            in_interests = random.randint(400, 800)
            in_data = random.randint(20, 80)
            in_nacks = random.randint(20, 60)
            in_satisfied = in_data
            in_timedout = in_interests - in_satisfied - in_nacks
            out_interests = in_interests
            out_data = in_data

        elif mode == 'Pulsing_IFA':
            # Alternate between zero/low and huge spikes
            if random.random() < 0.3: # Spike
                in_interests = random.randint(1500, 3000)
                in_data = random.randint(5, 20)
            else: # Low/Normal
                in_interests = random.randint(10, 50)
                in_data = int(in_interests * random.uniform(0.9, 1.0))
            in_nacks = random.randint(5, 50)
            in_satisfied = in_data
            in_timedout = in_interests - in_satisfied - in_nacks
            out_interests = in_interests
            out_data = in_data

        # Ensure no negative timeouts
        in_timedout = max(0, in_timedout)
        
        # Outgoing counts (Simplified for dummy generator)
        out_nacks = in_nacks
        out_satisfied = in_satisfied
        out_timedout = in_timedout
        
        row = {
            'Time': time_sec,
            'Node': node,
            'InInterests': in_interests,
            'InData': in_data,
            'InNacks': in_nacks,
            'InSatisfiedInterests': in_satisfied,
            'InTimedOutInterests': in_timedout,
            'OutInterests': out_interests,
            'OutData': out_data,
            'OutNacks': out_nacks,
            'OutSatisfiedInterests': out_satisfied,
            'OutTimedOutInterests': out_timedout,
            'Label': mode
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # -----------------------------------------------------
    # Feature Engineering (PART 4 requirements)
    # -----------------------------------------------------
    # Add small epsilon to prevent division by zero
    eps = 1e-6
    
    df['interest_rate'] = df['InInterests']
    df['data_rate'] = df['InData']
    df['satisfaction_ratio'] = df['InSatisfiedInterests'] / (df['InInterests'] + eps)
    df['timeout_ratio'] = df['InTimedOutInterests'] / (df['InInterests'] + eps)
    df['nack_ratio'] = df['InNacks'] / (df['InInterests'] + eps)
    # Estimate PIT Size: Interests that came in but haven't been satisfied or timed out or Nacked
    # For a snapshot, we just simulate a running PIT estimate based on timedout and nacks over interests
    df['pit_estimate'] = df['InInterests'] * df['timeout_ratio'] * random.uniform(1.0, 5.0)

    # Clean missing values (Just in case, though dummy data has none)
    df = df.fillna(0)
    
    # Save to CSV
    csv_path = os.path.join(DATA_DIR, 'ndn_traffic.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated dummy dataset with {num_samples} records at: {csv_path}")
    print("\nSample Preview:")
    print(df.head())

if __name__ == "__main__":
    generate_dummy_data()

import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

def run_experiment(args_dict):
    """
    Run a single experiment with the specified parameters by calling RL_assignment.py
    
    Args:
        args_dict (dict): Dictionary containing the parameters for the experiment
    
    Returns:
        dict: The parameters used and the directory where results are stored
    """            
    # Check if output directory already exists
    output_dir = args_dict["output_dir"]
    if os.path.exists(output_dir):
        print(f"Experiment results already exist in {output_dir}, skipping...")
        return args_dict
        
    # Convert dictionary to command-line arguments
    command = ["python", "RL_assignment.py"]
    
    for key, value in args_dict.items():
        command.append(f"--{key}")
        command.append(str(value))
    
    # Run the experiment
    print(f"Running experiment with: {' '.join(command)}")
    subprocess.run(command)
    
    # Return the configuration for logging
    return args_dict

def run_learning_rate_experiments(base_args):
    """
    Run experiments with different learning rates
    """
    print("\n=== RUNNING LEARNING RATE EXPERIMENTS ===")
    
    # Define learning rates to test
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    
    experiments = []
    for lr in learning_rates:
        args = base_args.copy()
        args["learning_rate"] = lr
        args["output_dir"] = f"./Results/logs_lr_{lr}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("learning_rate", [f"./Results/logs_lr_{lr}" for lr in learning_rates])

def run_batch_size_experiments(base_args):
    """
    Run experiments with different batch sizes
    """
    print("\n=== RUNNING BATCH SIZE EXPERIMENTS ===")
    
    # Define batch sizes to test
    batch_sizes = [64, 128, 256, 512]
    
    experiments = []
    for bs in batch_sizes:
        args = base_args.copy()
        args["batch_size"] = bs
        args["output_dir"] = f"./Results/logs_bs_{bs}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("batch_size", [f"./Results/logs_bs_{bs}" for bs in batch_sizes])

def run_chunk_size_experiments(base_args):
    """
    Run experiments with different episode lengths (chunk sizes)
    """
    print("\n=== RUNNING CHUNK SIZE EXPERIMENTS ===")
    
    # Define chunk sizes to test
    chunk_sizes = [50, 100, 200, 400]
    
    experiments = []
    for cs in chunk_sizes:
        args = base_args.copy()
        args["chunk_size"] = cs
        args["output_dir"] = f"./Results/logs_chunk_{cs}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("chunk_size", [f"./Results/logs_chunk_{cs}" for cs in chunk_sizes])

def run_algorithm_experiments(base_args):
    """
    Run experiments with different RL algorithms
    """
    print("\n=== RUNNING ALGORITHM EXPERIMENTS ===")
    
    # Define algorithms to test
    algorithms = ["SAC", "PPO", "TD3", "DDPG"]
    
    experiments = []
    for alg in algorithms:
        args = base_args.copy()
        args["model"] = alg
        args["output_dir"] = f"./Results/logs_alg_{alg}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("algorithm", [f"./Results/logs_alg_{alg}" for alg in algorithms])

def run_network_arch_experiments(base_args):
    """
    Run experiments with different network architectures
    """
    print("\n=== RUNNING NETWORK ARCHITECTURE EXPERIMENTS ===")
    
    # Define network architectures to test
    architectures = ["64,64", "128,128", "256,256", "512,512"]
    arch_names = ["small", "medium", "large", "xlarge"]
    
    experiments = []
    for arch, name in zip(architectures, arch_names):
        args = base_args.copy()
        args["net_arch"] = arch
        args["output_dir"] = f"./Results/logs_arch_{name}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("network_architecture", [f"./Results/logs_arch_{name}" for name in arch_names])

def run_gamma_experiments(base_args):
    """
    Run experiments with different discount factors (gamma)
    """
    print("\n=== RUNNING GAMMA EXPERIMENTS ===")
    
    # Define gamma values to test
    gamma_values = [0.9, 0.95, 0.99, 0.999]
    
    experiments = []
    for gamma in gamma_values:
        args = base_args.copy()
        args["gamma"] = gamma
        args["output_dir"] = f"./Results/logs_gamma_{gamma}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("gamma", [f"./Results/logs_gamma_{gamma}" for gamma in gamma_values])

def run_ent_coef_experiments(base_args):
    """
    Run experiments with different entropy coefficients (ent_coef)
    """
    print("\n=== RUNNING ENTROPY COEFFICIENT EXPERIMENTS ===")
    
    # Define entropy coefficient values to test
    ent_coef_values = [-1.0, 0.01, 0.05, 0.1]
    ent_coef_names = ["auto", "0.01", "0.05", "0.1"]
    
    experiments = []
    for ec, name in zip(ent_coef_values, ent_coef_names):
        args = base_args.copy()
        args["ent_coef"] = ec
        args["output_dir"] = f"./Results/logs_ent_coef_{name}"
        experiments.append(args)
    
    # Run experiments
    for exp in experiments:
        run_experiment(exp)
    
    # Compile results
    compile_results("entropy_coefficient", [f"./Results/logs_ent_coef_{name}" for name in ent_coef_names])

def compile_results(experiment_name, log_dirs):
    """
    Compile results from multiple experiments into comparative visualizations
    
    Args:
        experiment_name (str): Name of the experiment parameter being varied
        log_dirs (list): List of directories containing experiment results
    """
    print(f"Compiling results for {experiment_name} experiments...")
    
    # Create directory for comparative results
    results_dir = f"./Results/comparative_results_{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)

    #====================================================================================================
    # Plot 1: Metric omparative plot
    #====================================================================================================
    
    # Collect metrics from each experiment
    metrics_data = []
    params = []
    
    for log_dir in log_dirs:
        # Extract parameter from directory name
        param = log_dir.split('_')[-1]
        params.append(param)
        
        # OPTION 1: Try to find metrics files first
        metrics_files = [f for f in os.listdir(log_dir) if (f.endswith('metrics_chunk100.csv') or f.startswith('sac_acc_metrics_chunk'))]
        
        if metrics_files:
            metrics_file = os.path.join(log_dir, metrics_files[0])
            try:
                metrics_df = pd.read_csv(metrics_file)
                metrics_dict = dict(zip(metrics_df['metric'], metrics_df['value']))
                metrics_dict['param'] = param
                metrics_data.append(metrics_dict)
                print(f"Found metrics file in {log_dir}")
            except Exception as e:
                print(f"Error reading metrics from {metrics_file}: {e}")
        else:
            # OPTION 2: Look for training_log.csv instead
            training_log = os.path.join(log_dir, 'training_log.csv')
            if os.path.exists(training_log):
                try:
                    log_df = pd.read_csv(training_log)
                    
                    # Use the final row (latest timestep) for metrics
                    final_row = log_df.iloc[-1]
                    
                    # Convert training log metrics to the format expected by the rest of the function
                    metrics_dict = {
                        'MAE': final_row.get('average_speed_error', 0),
                        'Distance_MAE': final_row.get('average_distance_error', 0),
                        'Jerk_Mean': final_row.get('average_jerk', 0),
                        'Reward': final_row.get('average_reward', 0),
                        'param': param
                    }
                    
                    # Add additional metrics if available
                    for col in log_df.columns:
                        if col not in ['timestep', 'average_reward', 'average_speed_error', 'average_distance_error', 'average_jerk']:
                            metrics_dict[col] = final_row[col]
                    
                    metrics_data.append(metrics_dict)
                    print(f"Using training_log.csv in {log_dir}")
                except Exception as e:
                    print(f"Error reading training log from {training_log}: {e}")
            else:
                print(f"No metrics or training log found in {log_dir}")
    
    if not metrics_data:
        print("No metrics data found to compile")
        return
    
    # Create comparative plots
    # Updated metrics list based on what might be available in training logs
    metrics_of_interest = ['MAE', 'MSE', 'RMSE', 'Distance_MAE', 'Jerk_Mean', 'Jerk_Variance',
                          'Safety_Violations_Percent', 'Speed_Difference_MAE', 'Reward']
    
    # Filter metrics that actually exist in the data
    available_metrics = [m for m in metrics_of_interest if all(m in df for df in metrics_data)]
    
    if not available_metrics:
        print("No common metrics found across experiments")
        return
    
    # Bar plot comparing key metrics - IMPROVED LAYOUT
    n_metrics = len(available_metrics)
    rows = (n_metrics + 1) // 2  # Calculate number of rows needed
    
    # Adjust figure size for better visibility
    fig_width = 15
    fig_height = 5 * rows  # Scale height by number of rows
    
    plt.figure(figsize=(fig_width, fig_height))
    metrics_df = pd.DataFrame(metrics_data)
    
    for i, metric in enumerate(available_metrics):
        plt.subplot(rows, 2, i+1)
        
        # Improved bar plot
        x = np.arange(len(metrics_df['param']))
        bars = plt.bar(x, metrics_df[metric], width=0.6)
        
        # Set x-ticks with parameter values
        plt.xticks(x, metrics_df['param'])
        
        # Add titles and labels with better formatting
        plt.title(f"{metric} by {experiment_name}", fontsize=12, fontweight='bold')
        plt.ylabel(metric, fontsize=10)
        plt.xlabel(experiment_name, fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of each bar with better visibility
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + (height * 0.01),  # Position slightly above bar
                f'{height:.4f}',
                ha='center', 
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
    
    plt.tight_layout(pad=3.0)  # Increase padding for better spacing
    plt.subplots_adjust(hspace=0.4)  # Add more space between subplots
    
    plt.savefig(os.path.join(results_dir, f"comparison_metrics_{experiment_name}.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create summary table
    summary_df = pd.DataFrame(metrics_data)
    summary_file = os.path.join(results_dir, f"summary_{experiment_name}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Results compiled and saved to {results_dir}")
    
    #====================================================================================================
    # Plot 2: Speed profiles and following distances with improved error handling
    #====================================================================================================

    plt.figure(figsize=(15, 12))
    plot_generated = False
    
    for i, log_dir in enumerate(log_dirs[:4]):  # Limit to first 4 to avoid overcrowding
        # First try to find test_results files
        try:
            # OPTION 1: Look for test_results files
            results_files = [f for f in os.listdir(log_dir) if 'test_results' in f]
            
            if results_files:
                results_file = os.path.join(log_dir, results_files[0])
                results_df = pd.read_csv(results_file)
                param = log_dir.split('_')[-1]
                
                # Plot speed profile from test_results
                if all(col in results_df.columns for col in ['timestep', 'predicted_speed']):
                    max_rows = min(500, len(results_df))
                    plt.subplot(4, 2, i*2+1)
                    
                    if 'reference_speed' in results_df.columns:
                        plt.plot(results_df['timestep'][:max_rows], 
                                 results_df['reference_speed'][:max_rows], 
                                 label="Reference Speed", linestyle="--")
                    
                    if 'lead_vehicle_speed' in results_df.columns:
                        plt.plot(results_df['timestep'][:max_rows], 
                                 results_df['lead_vehicle_speed'][:max_rows], 
                                 label="Lead Vehicle", linestyle="-.")
                    
                    plt.plot(results_df['timestep'][:max_rows], 
                             results_df['predicted_speed'][:max_rows], 
                             label=f"Ego Vehicle ({param})")
                    
                    plt.ylabel("Speed (m/s)")
                    plt.title(f"{experiment_name}={param}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plot_generated = True
                    
                    # Plot distance if available
                    if 'distance_to_lead' in results_df.columns:
                        plt.subplot(4, 2, i*2+2)
                        plt.plot(results_df['timestep'][:max_rows], 
                                results_df['distance_to_lead'][:max_rows], 
                                label=f"Following Distance ({param})")
                        
                        # Add min/max distance lines if available
                        if 'min_distance' in results_df.columns and 'max_distance' in results_df.columns:
                            min_dist = results_df['min_distance'].iloc[0]
                            max_dist = results_df['max_distance'].iloc[0]
                            plt.axhline(y=min_dist, color='r', linestyle='-', alpha=0.3, label=f"Min ({min_dist}m)")
                            plt.axhline(y=max_dist, color='g', linestyle='-', alpha=0.3, label=f"Max ({max_dist}m)")
                        else:
                            # Use default values
                            plt.axhline(y=5, color='r', linestyle='-', alpha=0.3, label="Min (5m)")
                            plt.axhline(y=30, color='g', linestyle='-', alpha=0.3, label="Max (30m)")
                        
                        plt.ylabel("Distance (m)")
                        plt.title(f"Following Distance - {experiment_name}={param}")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
            else:
                # OPTION 2: Use training_log.csv for visualizations
                training_log = os.path.join(log_dir, 'training_log.csv')
                if os.path.exists(training_log):
                    log_df = pd.read_csv(training_log)
                    param = log_dir.split('_')[-1]
                    
                    if not log_df.empty:
                        # Plot training curves
                        plt.subplot(4, 2, i*2+1)
                        
                        # Plot speed error over time
                        if 'average_speed_error' in log_df.columns:
                            plt.plot(log_df['timestep'], log_df['average_speed_error'], 
                                    label=f"Speed Error ({param})")
                            plt.ylabel("Speed Error (m/s)")
                            plt.title(f"Speed Error - {experiment_name}={param}")
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plot_generated = True
                        
                        # Plot distance error
                        if 'average_distance_error' in log_df.columns:
                            plt.subplot(4, 2, i*2+2)
                            plt.plot(log_df['timestep'], log_df['average_distance_error'], 
                                    label=f"Distance Error ({param})")
                            plt.ylabel("Distance Error (m)")
                            plt.title(f"Distance Error - {experiment_name}={param}")
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                
        except Exception as e:
            print(f"Error processing files in {log_dir}: {e}")
            continue
    
    if plot_generated:
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"performance_profiles_{experiment_name}.png"), 
                    dpi=150, bbox_inches='tight')
    else:
        print(f"No data available to generate performance profiles for {experiment_name}")
    
    plt.close()
    
    #====================================================================================================
    # Plot 3: Additional ACC-specific plots if data is available (acceleration/jerk)
    #====================================================================================================

    plt.figure(figsize=(15, 12))
    curves_generated = False
    
    for i, log_dir in enumerate(log_dirs):
        training_log = os.path.join(log_dir, 'training_log.csv')
        if os.path.exists(training_log):
            try:
                log_df = pd.read_csv(training_log)
                param = log_dir.split('_')[-1]
                
                if not log_df.empty:
                    # Plot reward curve
                    plt.subplot(2, 2, 1)
                    plt.plot(log_df['timestep'], log_df['average_reward'], 
                             label=f"{param}")
                    curves_generated = True
                    
                    # Plot speed error
                    if 'average_speed_error' in log_df.columns:
                        plt.subplot(2, 2, 2)
                        plt.plot(log_df['timestep'], log_df['average_speed_error'], 
                                 label=f"{param}")
                    
                    # Plot distance error
                    if 'average_distance_error' in log_df.columns:
                        plt.subplot(2, 2, 3)
                        plt.plot(log_df['timestep'], log_df['average_distance_error'], 
                                 label=f"{param}")
                    
                    # Plot jerk
                    if 'average_jerk' in log_df.columns:
                        plt.subplot(2, 2, 4)
                        plt.plot(log_df['timestep'], log_df['average_jerk'], 
                                 label=f"{param}")
            except Exception as e:
                print(f"Error processing training log in {log_dir}: {e}")
                continue
    
    if curves_generated:
        plt.subplot(2, 2, 1)
        plt.title("Average Reward", fontsize=12, fontweight='bold')
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.title("Average Speed Error", fontsize=12, fontweight='bold')
        plt.xlabel("Timestep")
        plt.ylabel("Error (m/s)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.title("Average Distance Error", fontsize=12, fontweight='bold')
        plt.xlabel("Timestep")
        plt.ylabel("Error (m)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.title("Average Jerk", fontsize=12, fontweight='bold')
        plt.xlabel("Timestep")
        plt.ylabel("Jerk (m/s³)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"learning_curves_{experiment_name}.png"), 
                    dpi=150, bbox_inches='tight')
    
    plt.close()
    
def compile_final_report():
    """
    Create a final report compiling the best results from each experiment
    """
    print("Compiling final report...")
    
    # Create results directory
    final_dir = "./Results/final_report"
    os.makedirs(final_dir, exist_ok=True)
    
    # Find all comparative results directories
    comparative_dirs = [d for d in os.listdir("./Results") if d.startswith("comparative_results_")]
    
    if not comparative_dirs:
        print("No comparative results directories found. Run experiments first.")
        return
    
    # Extract best configuration for each experiment type
    best_configs = {}
    
    # Metrics to consider for finding "best" configuration
    # For ACC, we care about both speed following and distance maintenance
    key_metrics = ['MAE_Speed', 'MAE_Distance', 'Mean_Absolute_Jerk', 'Distance_In_Range_Percent']
    
    for dir_name in comparative_dirs:
        param_type = dir_name.replace("comparative_results_", "")
        summary_file = os.path.join("./Results", dir_name, f"summary_{param_type}.csv")
        
        if not os.path.exists(summary_file):
            print(f"Summary file not found: {summary_file}")
            continue
            
        try:
            # Load summary data
            summary_df = pd.read_csv(summary_file)
            
            if 'param' not in summary_df.columns:
                print(f"'param' column missing in {summary_file}")
                continue
            
            # Check which metrics are available in this summary
            available_key_metrics = [m for m in key_metrics if m in summary_df.columns]
            
            if not available_key_metrics:
                print(f"No key metrics found in {summary_file}")
                continue
            
            # First prioritize safety - configurations with fewer safety violations
            if 'Safety_Violations_Percent' in summary_df.columns:
                # Filter to configurations with lowest safety violations
                min_violations = summary_df['Safety_Violations_Percent'].min()
                safe_configs = summary_df[summary_df['Safety_Violations_Percent'] <= min_violations * 1.2]  # Allow 20% margin
                
                if not safe_configs.empty:
                    summary_df = safe_configs
            
            # Find best configuration based on combined metrics
            # We'll normalize and weight different metrics
            if len(available_key_metrics) > 1:
                # Normalize each metric to 0-1 range
                for metric in available_key_metrics:
                    min_val = summary_df[metric].min()
                    max_val = summary_df[metric].max()
                    range_val = max_val - min_val if max_val > min_val else 1.0
                    
                    # Avoid division by zero
                    if range_val > 0:
                        summary_df[f'norm_{metric}'] = (summary_df[metric] - min_val) / range_val
                    else:
                        summary_df[f'norm_{metric}'] = 0
                
                # Define metrics and their directions (True if higher is better, False if lower is better)
                metrics_direction = {
                    'Mean_Speed_Diff': False,   # Lower is better
                    'Mean_Absolute_Jerk': False, # Lower is better
                    'Jerk_Variance': False,     # Lower is better
                    'Distance_In_Range_Percent': True  # Higher is better
                }

                # Assign weights to each metric
                weights = {}
                if 'Mean_Speed_Diff' in available_key_metrics:
                    weights['Mean_Speed_Diff'] = 0.25
                if 'Mean_Absolute_Jerk' in available_key_metrics:
                    weights['Mean_Absolute_Jerk'] = 0.25
                if 'Jerk_Variance' in available_key_metrics:
                    weights['Jerk_Variance'] = 0.2
                if 'Distance_In_Range_Percent' in available_key_metrics:
                    weights['Distance_In_Range_Percent'] = 0.3

                # Normalize weights
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v/total_weight for k, v in weights.items()}

                # Normalize each metric to 0-1 range, accounting for whether higher or lower is better
                for metric in weights.keys():
                    if metric in summary_df.columns:
                        min_val = summary_df[metric].min()
                        max_val = summary_df[metric].max()
                        range_val = max_val - min_val if max_val > min_val else 1.0
                        
                        if range_val > 0:
                            if metric in metrics_direction and metrics_direction[metric]:
                                # For metrics where higher is better (like Distance_In_Range_Percent)
                                # Invert the normalization so higher values = lower scores (better)
                                summary_df[f'norm_{metric}'] = (max_val - summary_df[metric]) / range_val
                            else:
                                # For metrics where lower is better
                                summary_df[f'norm_{metric}'] = (summary_df[metric] - min_val) / range_val
                        else:
                            summary_df[f'norm_{metric}'] = 0

                # Calculate weighted scores
                summary_df['combined_score'] = 0
                for metric in weights.keys():
                    if metric in summary_df.columns:
                        summary_df['combined_score'] += weights[metric] * summary_df[f'norm_{metric}']
                
                best_idx = summary_df['combined_score'].idxmin()
                best_param = summary_df.loc[best_idx, 'param']
                
                # Store all relevant metrics for the best config
                metrics_dict = {metric: summary_df.loc[best_idx, metric] for metric in available_key_metrics}
                best_configs[param_type] = (best_param, metrics_dict)
            
            # Fallback to first available key metric if combined metrics aren't available
            elif available_key_metrics:
                primary_metric = available_key_metrics[0]
                best_idx = summary_df[primary_metric].idxmin()
                best_param = summary_df.loc[best_idx, 'param']
                best_value = summary_df.loc[best_idx, primary_metric]
                best_configs[param_type] = (best_param, {primary_metric: best_value})
                
        except Exception as e:
            print(f"Error processing {summary_file}: {e}")
            continue
    
    # Create summary table of best configurations
    if best_configs:
        best_rows = []
        for param_type, (param_val, metrics) in best_configs.items():
            row = {"Parameter": param_type, "Best Value": param_val}
            row.update(metrics)
            best_rows.append(row)
            
        best_df = pd.DataFrame(best_rows)
        best_file = os.path.join(final_dir, "best_configurations.csv")
        best_df.to_csv(best_file, index=False)
        print(f"Best configurations saved to {best_file}")
        
        # Create visualization of best configurations for key metrics
        available_metrics = set()
        for _, metrics_dict in best_configs.values():
            available_metrics.update(metrics_dict.keys())
        
        metrics_to_plot = [m for m in key_metrics if m in available_metrics]
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(12, 7))
            
            # Gather data for this metric
            params = []
            values = []
            
            for param_type, (param_val, metrics_dict) in best_configs.items():
                if metric in metrics_dict:
                    params.append(f"{param_type}\n({param_val})")
                    values.append(metrics_dict[metric])
            
            if not params:
                continue
                
            # Create bar chart with improved formatting
            x = np.arange(len(params))
            bars = plt.bar(x, values, width=0.6)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + (height * 0.01),
                    f'{height:.4f}',
                    ha='center', 
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )
            
            plt.title(f"Best {metric} by Parameter Type", fontsize=14, fontweight='bold')
            plt.ylabel(metric, fontsize=12)
            plt.xticks(x, params, fontsize=10)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save the figure
            metric_fig_path = os.path.join(final_dir, f"best_{metric}_by_parameter.png")
            plt.savefig(metric_fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Created metric comparison chart: {metric_fig_path}")
    
        # Create combined chart with all key metrics
        try:
            if metrics_to_plot:
                plt.figure(figsize=(15, 10))
                
                # Prepare data for grouped bar chart
                param_types = list(best_configs.keys())
                
                # Create a modified list of metrics for the chart
                chart_metrics = metrics_to_plot.copy()
                
                # Replace Distance_In_Range_Percent with Distance_Out_Of_Range_Percent for the chart only
                if 'Distance_In_Range_Percent' in chart_metrics:
                    chart_metrics.remove('Distance_In_Range_Percent')
                    chart_metrics.append('Distance_Out_Of_Range_Percent')
                
                metrics_data = {metric: [] for metric in chart_metrics}
                
                for param_type in param_types:
                    _, metrics_dict = best_configs[param_type]
                    for metric in chart_metrics:
                        if metric == 'Distance_Out_Of_Range_Percent':
                            # Transform the in-range to out-of-range (100% - in_range%)
                            if 'Distance_In_Range_Percent' in metrics_dict:
                                out_of_range = 100.0 - metrics_dict['Distance_In_Range_Percent']
                                metrics_data[metric].append(out_of_range)
                            else:
                                metrics_data[metric].append(0)
                        elif metric in metrics_dict:
                            metrics_data[metric].append(metrics_dict[metric])
                        else:
                            metrics_data[metric].append(0)
                
                # Set width of bars
                barWidth = 0.8 / len(chart_metrics) if len(chart_metrics) > 0 else 0.4
                
                # Set position of bars on X axis
                positions = []
                for i in range(len(chart_metrics)):
                    positions.append([x + barWidth * i for x in range(len(param_types))])
                
                # Make the plot
                for i, metric in enumerate(chart_metrics):
                    plt.bar(positions[i], 
                            metrics_data[metric], 
                            width=barWidth, 
                            label=metric if metric != 'Distance_Out_Of_Range_Percent' else 'Distance_Out_Of_Range_Percent (%)')
                
                # Add labels, title and legend
                plt.xlabel('Parameter Type', fontsize=12)
                plt.ylabel('Metric Value (Lower is Better)', fontsize=12)
                plt.title('Best Configurations Comparison by Metric', fontsize=14, fontweight='bold')
                plt.xticks([r + barWidth * (len(chart_metrics) - 1) / 2 for r in range(len(param_types))], 
                        [f"{pt}\n({best_configs[pt][0]})" for pt in param_types])
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                # Save the figure
                combined_fig_path = os.path.join(final_dir, "combined_metrics_comparison.png")
                plt.savefig(combined_fig_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Created combined metrics chart: {combined_fig_path}")
            else:
                print("No metrics available for combined chart")
        except Exception as e:
            print(f"Error creating combined metrics chart: {e}")
        
        # Create summary visualization
        plt.figure(figsize=(12, 8))
        
        # Create a table visualization of best configurations
        table_data = []
        columns = ["Parameter", "Best Value"] + metrics_to_plot

        for param_type, (param_val, metrics_dict) in best_configs.items():
            row = [param_type, param_val]
            for metric in metrics_to_plot:
                # Format the metric value to 4 decimal places if it's a number
                if metric in metrics_dict:
                    try:
                        value = float(metrics_dict[metric])
                        formatted_value = f"{value:.4f}"  # Format to 4 decimal places
                    except (ValueError, TypeError):
                        formatted_value = metrics_dict.get(metric, "N/A")
                else:
                    formatted_value = "N/A"
                row.append(formatted_value)
            table_data.append(row)
        
        # Create the table
        plt.axis('off')  # Turn off axis
        table = plt.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add title
        plt.title("Best Configurations Summary", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save the table visualization
        table_fig_path = os.path.join(final_dir, "best_configurations_table.png")
        plt.savefig(table_fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Created summary table visualization: {table_fig_path}")
    
    print(f"Final report compilation completed in {final_dir}")

def run_full_dataset_test(base_args):
    """
    Run a test on the full 1200-step dataset using the best configuration
    """
    print("Running test on full 1200-step dataset...")
    
    # Check if best model exists
    if not os.path.exists("./best_model"):
        print("No best model found. Running best configuration experiment first...")
        run_best_configuration_experiment(base_args)
    
    # Use the best model to test on full dataset
    command = ["python", "RL_assignment.py",
               "--test_only", "True",
               "--model_path", "./best_model",
               "--output_dir", "./full_dataset_results",
               "--enable_acc", "True",  # Ensure ACC is enabled
               "--test_full_dataset", "True"]  # Test on the full 1200-step dataset
    
    print(f"Running full dataset test with: {' '.join(command)}")
    subprocess.run(command)
    
    print("Full dataset test completed. See ./full_dataset_results for test output.")

def run_best_configuration_experiment(base_args):
    """
    Run an experiment with the best configuration found from previous experiments
    """
    # Load best configurations
    best_config_file = "./final_report/best_configurations.csv"
    
    if not os.path.exists(best_config_file):
        print("No best configuration file found. Run all experiments first.")
        return
        
    best_df = pd.read_csv(best_config_file)
    
    # Update with best configurations
    for _, row in best_df.iterrows():
        param = row['Parameter']
        value = row['Best Value']
        
        # Handle different parameter types
        if param == "learning_rate":
            try:
                base_args[param] = float(value)
            except:
                pass
        elif param == "batch_size":
            try:
                base_args[param] = int(value)
            except:
                pass
        elif param == "chunk_size":
            try:
                base_args[param] = int(value)
            except:
                pass
        elif param == "network_architecture":
            if value == "small":
                base_args["net_arch"] = "64,64"
            elif value == "medium":
                base_args["net_arch"] = "128,128"
            elif value == "large":
                base_args["net_arch"] = "256,256"
            elif value == "xlarge":
                base_args["net_arch"] = "512,512"
        elif param == "algorithm":
            base_args["model"] = value
        elif param == "gamma":
            try:
                base_args[param] = float(value)
            except:
                pass
        elif param == "entropy_coefficient":
            if value == "auto":
                base_args["ent_coef"] = -1.0
            else:
                try:
                    base_args["ent_coef"] = float(value)
                except:
                    pass
    
    # Run the experiment with best configuration
    base_args["output_dir"] = "./best_model"
    run_experiment(base_args)
    
    print("Best configuration experiment completed.")

def main():
    parser = argparse.ArgumentParser(description="Run RL experiments for Adaptive Cruise Control")
    parser.add_argument("--experiment", type=str, default="all",
                      choices=["all", "learning_rate", "batch_size", "chunk_size", 
                               "algorithm", "network_arch", "gamma", "ent_coef"],
                      help="Type of experiment to run")
    parser.add_argument("--timesteps", type=int, default=1200,
                      help="Total timesteps for training")
    parser.add_argument("--test_full", action="store_true",
                      help="Run test on full 1200-step dataset")
    
    args = parser.parse_args()
    
    # Base configuration used for all experiments - updated for ACC
    base_args = {
        "model": "SAC",              # Default algorithm
        "learning_rate": 1e-4,
        "batch_size": 256,
        "buffer_size": 200000,
        "chunk_size": 100,
        "gamma": 0.99,
        "tau": 0.005,
        "ent_coef": -1.0,            # 'auto'
        "net_arch": "256,256",
        "total_timesteps": args.timesteps
    }
    
    start_time = time.time()
    
    # Run selected experiment
    if args.experiment == "learning_rate" or args.experiment == "all":
        run_learning_rate_experiments(base_args)
        
    if args.experiment == "batch_size" or args.experiment == "all":
        run_batch_size_experiments(base_args)
        
    if args.experiment == "chunk_size" or args.experiment == "all":
        run_chunk_size_experiments(base_args)
        
    if args.experiment == "algorithm" or args.experiment == "all":
        run_algorithm_experiments(base_args)
        
    if args.experiment == "network_arch" or args.experiment == "all":
        run_network_arch_experiments(base_args)
        
    if args.experiment == "gamma" or args.experiment == "all":
        run_gamma_experiments(base_args)
        
    if args.experiment == "ent_coef" or args.experiment == "all":
        run_ent_coef_experiments(base_args)


    end_time = time.time()
    print(f"\nAll experiments completed in {end_time - start_time:.2f} seconds")

    # Generate final compilation of all results if all experiments were run
    if args.experiment == "all":
        print("\n=== GENERATING FINAL REPORT ===")
        compile_final_report()
        
    # Run test on full dataset if requested
    if args.test_full:
        print("\n=== RUNNING FULL DATASET TEST ===")
        run_full_dataset_test(base_args)
        
if __name__ == "__main__":
    main()
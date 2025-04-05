import os
import json
import traceback
from __eval import LLMCodeEvaluator

def run_evaluation(inf_dir: str, n_completions: int):
    """
    Evaluate model predictions using the LLMCodeEvaluator class.
    
    Args:
        inf_dir (str): Directory containing predictions.json
        
    Returns:
        tuple: (metrics, summary)
    """
    print(f"Starting evaluation on predictions in {inf_dir}")
    
    # Initialize the evaluator
    evaluator = LLMCodeEvaluator()
    
    try:
        # Run the evaluation pipeline
        metrics, summary = evaluator.evaluate_and_summarize(inf_dir, n_completions=n_completions)

        # Save the evaluation results
        with open(os.path.join(inf_dir, "evaluation.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        with open(os.path.join(inf_dir, "detailed_metrics.json"), "w") as f:
            # Convert any non-serializable values to strings
            serializable_metrics = []
            for metric in metrics:
                serializable_metric = {}
                for k, v in metric.items():
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        serializable_metric[k] = v
                    else:
                        serializable_metric[k] = str(v)
                serializable_metrics.append(serializable_metric)
            
            # Write the entire list once, outside the loop
            json.dump(serializable_metrics, f, indent=2) 
            
        print(f"Evaluation complete. Results saved to {inf_dir}/evaluation.json")
        return metrics, summary
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Path to the directory containing predictions.json
    inf_dir = "results/length/deepseekcoder/inference/20250405_0048"   #"results/length/CodeLlama/inference/20250324_1930" #"/ceph/pratz/GitHub_repos/master-thesis/results/length/CodeLlama_20250316_2016/inference/20250324_2025" # "results/length/CodeLlama_20250314_1803_test/inference"  
    
    # Run evaluation
    metrics, summary = run_evaluation(inf_dir, n_completions=3)
    
    # Check if evaluation was successful
    if metrics is not None and summary is not None:
        print(f"Evaluation completed successfully!")
        print(f"Number of evaluated samples: {len(metrics)}")
        print(f"Summary metrics: {summary}")
    else:
        print("Evaluation failed. Check the error messages above.")
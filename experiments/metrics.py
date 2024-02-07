import sklearn.metrics as skm

def fidelity(lstm_preds, tree_preds):

  fidelity = skm.accuracy_score(lstm_preds, tree_preds)
  print(f"Fidelity: {fidelity:.3f}")
  return fidelity

def tree_node_depth(decision_tree):
  # Compute the interpretability metric
  depth = decision_tree.tree_.max_depth
  n_nodes = decision_tree.tree_.node_count
  print(f"Depth: {depth}")
  print(f"Number of nodes: {n_nodes}")
  
  return depth, n_nodes
  
  
def objective_evaluation(tree_model, lstm_preds, tree_preds):
  # Compute fidelity and tree node depth
  fidelity_score = fidelity(lstm_preds, tree_preds)
  depth, n_nodes = tree_node_depth(tree_model)
  
  return fidelity_score, depth, n_nodes
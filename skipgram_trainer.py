"""
Skip-Gram with Negative Sampling (SGNS) for Network Embeddings

Implements Skip-Gram with Negative Sampling to learn embeddings from text networks.
Includes training, evaluation, and visualization tools.

KEY FEATURES:
1. Filters punctuation tokens to prevent hub poisoning
2. Proper negative sampling (5-20 negatives per positive)
3. Weighted sampling by co-occurrence frequency
4. Anti-overfitting: dropout, weight decay, label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import networkx as nx
import requests
import zipfile
import json
import os
from typing import List, Dict, Set, Tuple

import unittest
from collections import Counter


# ============================================================================
# Utilities
# ============================================================================

def download_file(url, out_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {out_path}")


def prepare_visual_genome_text(zip_url, zip_path="region_descriptions.json.zip", 
                                json_path="region_descriptions.json",
                                output_path="vg_text.txt"):
    """Download, unzip, and process Visual Genome region descriptions."""
    
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping processing.")
        return output_path

    if not os.path.exists(zip_path):
        download_file(zip_url, zip_path)
    
    if not os.path.exists(json_path):
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    print(f"Processing {json_path} into {output_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    phrases = [region['phrase'] for img in data for region in img['regions']]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(" . ".join(phrases))
    
    print(f"Processed {len(phrases):,} phrases into {output_path}")
    return output_path


def filter_punctuation_from_network(network_data, punctuation_tokens={'.', ',', '<RARE>', "'"}):
    """
    Remove punctuation tokens from network to prevent hub poisoning.
    
    Punctuation creates massive hubs that bridge unrelated sentences,
    poisoning the graph structure and making embeddings meaningless.
    """
    original_graph = network_data['graph']
    original_nodes = network_data['nodes']
    original_distance_matrix = network_data['distance_matrix']
    
    # Filter nodes
    filtered_nodes = [n for n in original_nodes if n not in punctuation_tokens]
    old_indices = [i for i, n in enumerate(original_nodes) if n not in punctuation_tokens]
    
    # Filter matrices
    filtered_distance_matrix = original_distance_matrix[np.ix_(old_indices, old_indices)]
    
    # Create filtered graph
    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(filtered_nodes)
    for u, v in original_graph.edges():
        if u in filtered_nodes and v in filtered_nodes:
            filtered_graph.add_edge(u, v)
    
    print(f"\n PUNCTUATION FILTER:")
    print(f"  Removed: {punctuation_tokens}")
    print(f"  Nodes: {len(original_nodes):,} -> {len(filtered_nodes):,}")
    print(f"  Edges: {original_graph.number_of_edges():,} -> {filtered_graph.number_of_edges():,}")
    
    return {
        **network_data,
        'graph': filtered_graph,
        'nodes': filtered_nodes,
        'distance_matrix': filtered_distance_matrix
    }


# ============================================================================
# Dataset
# ============================================================================

class SkipGramDataset(torch.utils.data.Dataset):
    """
    Skip-Gram dataset for learning node embeddings from a graph structure.
    
    This dataset:
    - Builds (center, context) training pairs from graph neighbors
    - Computes importance weights for weighted sampling
    - Samples negative examples on-the-fly during training
    - Handles multi-worker data loading with independent random streams
    
    Example Usage:
        >>> graph = nx.karate_club_graph()
        >>> nodes = list(graph.nodes())
        >>> dist_matrix = compute_distance_matrix(graph, nodes)  # your function
        >>> dataset = SkipGramDataset(graph, nodes, dist_matrix)
        >>> center, context, negatives = dataset[0]
        >>> print(f"Center: {center}, Context: {context}, Negatives: {negatives[:3]}...")
    """

    def __init__(
        self,
        graph: nx.Graph,
        nodes: List[str],
        distance_matrix: np.ndarray,
        num_negative: int = 15,
        context_size: int = 1,
    ):
        """
        Initialize the Skip-Gram dataset.
        
        Args:
            graph: NetworkX graph where nodes are tokens/words
            nodes: Ordered list of node labels (vocabulary)
            distance_matrix: Precomputed distances between nodes, shape (V, V)
            num_negative: Number of negative samples per positive pair (typically 5-20)
            context_size: Context radius in graph hops (1 = immediate neighbors)
        """
        super().__init__()
        
        # Store basic references
        self.graph = graph
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.vocab_size = len(nodes)
        self.num_negative = num_negative
        self.distance_matrix = distance_matrix
        
        # Step 1: Build context sets for each node
        # WHY: We need to know which nodes are "related" to create positive pairs
        self.contexts = self._build_contexts(context_size)
        
        # Step 2: Convert contexts into training pairs and compute weights
        # WHY: PyTorch needs explicit (center, context) pairs, and weighting helps
        #      the model focus on more important relationships
        self.pairs, self.weights = self._generate_weighted_pairs()
        
        # Step 3: Initialize per-worker RNG (lazily, in __getitem__)
        # WHY: Multi-worker DataLoaders need independent random streams
        self._local_rng = None
        
        # Print summary statistics
        self._print_stats()

    def _build_contexts(self, context_size: int) -> Dict[str, Set[str]]:
        contexts = {}
        # Iterate through all nodes in vocabulary
        for node in self.nodes:
            # Handle nodes not in graph (give them an empty set)
            if node not in self.graph:
                contexts[node] = set()
                continue
            
            # Compute shortest paths within cutoff using NetworkX
            path_lengths = nx.single_source_shortest_path_length(
                self.graph, 
                node, 
                cutoff=context_size
            )
            
            # Filter to valid vocabulary nodes with distance > 0 (exclude self)
            context_set = {
                neighbor 
                for neighbor, distance in path_lengths.items() 
                if distance > 0 and neighbor in self.node_to_idx
            }
            
            contexts[node] = context_set
        
        return contexts
            



    def _generate_weighted_pairs(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        pairs = []
        raw_distances = []
        # Iterate through all nodes and their contexts (from _build_contexts)
        for center_node, context_set in self.contexts.items():
            # Convert center word string to index
            center_idx = self.node_to_idx[center_node]
            
            # For each context word in this node's neighborhood
            for context_node in context_set:
                # Convert context word string to index
                context_idx = self.node_to_idx[context_node]
                
                # Store the (center, context) pair
                pairs.append((center_idx, context_idx))
                
                # Look up the distance from the precomputed matrix
                distance = self.distance_matrix[center_idx, context_idx]
                raw_distances.append(distance)

        # Edge case: if no pairs exist, return empty arrays
        if len(pairs) == 0:
            return [], np.array([], dtype=np.float32)

        raw_distances = np.array(raw_distances, dtype=np.float32)

        max_distance = np.max(raw_distances)
        weights = np.array([(max_distance + 1) - d for d in raw_distances], dtype=np.float32)
        weights = np.maximum(weights, 1e-6)
        weights = np.sqrt(weights)
        clip_threshold = np.percentile(weights, 95) * 3
        weights = np.clip(weights, a_min=None, a_max=clip_threshold)
        weights = (weights / weights.sum()) * len(weights)
        
        return pairs, weights

    def __getitem__(self, idx: int) -> Tuple[np.int64, np.int64, np.ndarray]:
        if self._local_rng is None:
            # Get worker info for multi-processing
            worker_info = torch.utils.data.get_worker_info()
            
            if worker_info is None:
                # Single-process mode: use a fixed seed
                seed = torch.initial_seed() % (2**32)
            else:
                # Multi-worker mode: unique seed per worker
                seed = (torch.initial_seed() + worker_info.id) % (2**32)
            
            # Create numpy RNG with unique seed
            self._local_rng = np.random.RandomState(seed)

        center_idx, context_idx = self.pairs[idx]
        center_node = self.nodes[center_idx]
        excluded = {
            self.node_to_idx[neighbor] 
            for neighbor in self.contexts[center_node]
            if neighbor in self.node_to_idx
        }
        excluded.add(center_idx)

        available = np.array(list(set(range(self.vocab_size)) - excluded), dtype=np.int64)
        if len(available) == 0:
            available = np.arange(self.vocab_size, dtype=np.int64)

        replace = len(available) < self.num_negative

        # Sample negative examples
        negatives = self._local_rng.choice(
            available,
            size=self.num_negative,
            replace=replace
        )
        
        return (
            np.int64(center_idx),
            np.int64(context_idx),
            negatives.astype(np.int64)
        )

    # ========================================================================
    # PROVIDED HELPER METHODS (no changes needed)
    # ========================================================================

    def get_sample_weights(self) -> np.ndarray:
        """
        Return per-pair weights for WeightedRandomSampler.
        
        Usage:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=dataset.get_sample_weights(),
                num_samples=len(dataset),
                replacement=True
            )
            loader = DataLoader(dataset, sampler=sampler, batch_size=32)
        """
        return self.weights

    def __len__(self) -> int:
        """Number of positive training pairs."""
        return len(self.pairs)

    def _print_stats(self):
        """Print dataset statistics for debugging."""
        print("\n SkipGramDataset Statistics:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Positive pairs: {len(self.pairs):,}")
        print(f"  Negatives per positive: {self.num_negative}")
        print(f"  Total samples per epoch: {len(self.pairs) * (1 + self.num_negative):,}")
        
        if self.weights.size > 0:
            print(f"\n  Weight distribution:")
            print(f"    Min: {self.weights.min():.6f}")
            print(f"    Mean: {self.weights.mean():.6f}")
            print(f"    Median: {np.median(self.weights):.6f}")
            print(f"    Max: {self.weights.max():.6f}")
        else:
            print("  [!]  No pairs found - check your graph and nodes!")


# ============================================================================
# Model
# ============================================================================

class SkipGramModel(nn.Module):
    """
    Skip-Gram model with Negative Sampling (SGNS).
    
    Architecture:
        - center_embeddings: Embedding(V, D) - represents words as query vectors
        - context_embeddings: Embedding(V, D) - represents words as key vectors
        - dropout: Regularization applied to center embeddings
    
    Why two embedding matrices?
        In Skip-Gram, words play two roles:
        1. As CENTER: "What contexts does this word appear in?"
        2. As CONTEXT: "What centers is this word a context for?"
        
        These are asymmetric relationships. Using separate embeddings lets the
        model learn different representations for each role, improving quality.
    
    Training objective:
        Maximize: P(context | center) for true pairs
        Minimize: P(negative | center) for random pairs
        
    Example:
        >>> model = SkipGramModel(vocab_size=1000, embedding_dim=128)
        >>> center = torch.tensor([5, 10])      # batch of 2 center words
        >>> context = torch.tensor([8, 15])     # their true contexts
        >>> negatives = torch.randint(0, 1000, (2, 10))  # 10 negatives each
        >>> loss = model(center, context, negatives)
        >>> print(loss.shape)  # torch.Size([2]) - loss per example
    """

    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float = 0.3):
        """
        Initialize Skip-Gram model.
        
        Args:
            vocab_size: Size of vocabulary (number of unique nodes/words)
            embedding_dim: Dimensionality of embedding vectors (typically 50-300)
            dropout: Dropout probability for regularization (prevents overfitting)
        """
        super().__init__()
        
        # Two embedding matrices: one for center words, one for context words
        # WHY: Asymmetric roles in Skip-Gram (see class docstring)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Dropout for regularization (applied only to center embeddings during training)
        # WHY: Prevents model from memorizing training pairs, improves generalization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings with proper scaling
        self._init_embeddings()

    def _init_embeddings(self):
        embedding_dim = self.center_embeddings.embedding_dim
        scale = 0.5 / embedding_dim
    
        nn.init.uniform_(self.center_embeddings.weight, -scale, scale)

        nn.init.uniform_(self.context_embeddings.weight, -scale, scale)
 
    def forward(
        self, 
        center: torch.Tensor,      # shape: (batch_size,)
        context: torch.Tensor,     # shape: (batch_size,)
        negatives: torch.Tensor,   # shape: (batch_size, num_negatives)
        apply_dropout: bool = True,
        label_smoothing: float = 0.1
    ) -> torch.Tensor:
        """
        Compute Skip-Gram Negative Sampling loss.
        
        Algorithm:
        1. Look up embeddings for center, context, and negative words
        2. Compute positive score: similarity(center, context)
        3. Compute negative scores: similarity(center, each negative)
        4. Apply label smoothing to targets (anti-overfitting)
        5. Compute binary cross-entropy loss using log-sigmoid
        6. Return negative loss (we'll minimize this, which maximizes log-likelihood)
        
        Mathematical formulation:
            Positive loss: -log(sigma(center * context))
            Negative loss: -Sum log(sigma(-center * negative_i))
            
            With label smoothing (alpha = 0.1):
            - True positive target: 0.9 instead of 1.0
            - True negative target: 0.9 instead of 1.0
            This prevents overconfident predictions
        
        Args:
            center: Batch of center word indices, shape (B,)
            context: Batch of true context word indices, shape (B,)
            negatives: Batch of negative word indices, shape (B, K)
            apply_dropout: Whether to apply dropout to center embeddings
            label_smoothing: Smoothing factor (0 = no smoothing, 0.1 = mild)
            
        Returns:
            loss: Per-example loss, shape (B,). Caller typically does loss.mean()
        
        HINTS:
        - Use self.center_embeddings(center) to look up embeddings
        - Dot product: torch.sum(a * b, dim=1) for element-wise mult + sum
        - Batch matrix multiply: torch.bmm(A, B) where A is (B,K,D), B is (B,D,1)
        - Log-sigmoid: F.logsigmoid(x) computes log(1/(1+exp(-x))) stably
        - Label smoothing formula: smoothed_target = (1 - alpha) for positive
        """
        
        if apply_dropout:
            center_emb = self.dropout(self.center_embeddings(center))
        else:
            center_emb = self.center_embeddings(center)

        context_emb = self.context_embeddings(context)

        negative_emb = self.context_embeddings(negatives)

        pos_score = torch.sum(center_emb * context_emb, dim=1)

        neg_score = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)
  

            # Positive loss
        pos_target = 1.0 - label_smoothing
        pos_loss = (pos_target * F.logsigmoid(pos_score) +
                     (1 - pos_target) * F.logsigmoid(-pos_score))
        
        # Negative loss
        neg_target = label_smoothing
        neg_loss = (neg_target * F.logsigmoid(neg_score) +
                     (1 - neg_target) * F.logsigmoid(-neg_score))
        
        # IMPORTANT: mean over negatives (prevents collapse)
        # neg_loss = neg_loss.mean(dim=1)
        neg_loss = neg_loss.sum(dim=1)
        
        return -(pos_loss + neg_loss)


    def get_embeddings(self) -> np.ndarray:
        """
        Extract the learned center embeddings as a numpy array.
        
        Why center embeddings?
            Both center and context embeddings contain learned information, but:
            - Center embeddings are what we optimized as "query" vectors
            - They're used during training with dropout (more robust)
            - Convention: use center embeddings for downstream tasks
            
        Alternative: You could average center + context embeddings, but this
        is less common and may not improve quality.
        
        Returns:
            embeddings: numpy array of shape (vocab_size, embedding_dim)
        
        """
        center_emb = self.center_embeddings.weight
        if center_emb.device.type != 'cpu':
            center_emb = center_emb.detach().cpu().numpy()
        else:
            center_emb = center_emb.detach().numpy()
        
        return center_emb




# ============================================================================
# Training
# ============================================================================

def train_embeddings(
    network_data,
    embedding_dim=128,
    batch_size=512,
    epochs=20,
    learning_rate=0.001,
    num_negative=15,
    validation_fraction=0.05,
    context_size=1,
    dropout=0.3,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=3,
    device=None,
    save_plot=True
):
    """
    Train Skip-Gram embeddings with weighted sampling.
    
    Args:
        network_data: Dict with 'graph', 'nodes', 'distance_matrix'
        embedding_dim: Embedding dimensionality
        batch_size: Training batch size
        epochs: Maximum epochs
        learning_rate: Initial learning rate
        num_negative: Negatives per positive (5-20 recommended)
        validation_fraction: Fraction for validation
        context_size: Graph distance for context (1=neighbors)
        dropout: Dropout rate (default: 0.3)
        weight_decay: L2 regularization (default: 1e-4)
        label_smoothing: Label smoothing factor (default: 0.1)
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
        save_plot: Save training curve
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Filter punctuation
    network_data = filter_punctuation_from_network(network_data)
    nodes = network_data['nodes']
    graph = network_data['graph']
    distance_matrix = network_data['distance_matrix']
    
    # Split edges
    all_edges = list(graph.edges())
    np.random.shuffle(all_edges)
    split_idx = int(len(all_edges) * (1 - validation_fraction))
    
    train_graph = nx.Graph()
    train_graph.add_nodes_from(nodes)
    train_graph.add_edges_from(all_edges[:split_idx])
    
    val_graph = nx.Graph()
    val_graph.add_nodes_from(nodes)
    val_graph.add_edges_from(all_edges[split_idx:])
    
    print(f"\nTrain edges: {len(all_edges[:split_idx]):,}, Val edges: {len(all_edges[split_idx:]):,}")
    
    # Create datasets
    train_dataset = SkipGramDataset(train_graph, nodes, distance_matrix, num_negative, context_size)
    val_dataset = SkipGramDataset(val_graph, nodes, distance_matrix, num_negative, context_size)
    
    # Create loaders with weighted sampling
    sampler = WeightedRandomSampler(
        weights=train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Initialize model
    model = SkipGramModel(len(nodes), embedding_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\nTraining on {device}")
    print(f"Vocab: {len(nodes)}, Embed dim: {embedding_dim}, Context: {context_size}, Negatives: {num_negative}")
    print(f"Regularization: dropout={dropout}, weight_decay={weight_decay}, label_smoothing={label_smoothing}")
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for i, (centers, contexts, negs) in train_pbar:
            centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
            
            loss = model(centers, contexts, negs, True, label_smoothing).mean()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #### control the gradeint
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'train_loss': f'{total_loss / (i + 1):.4f}'})
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        total_val_loss = 0.0
        
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=False)
        with torch.no_grad(): ## says do not change gradient stops the models upadtes.
            for i, (centers, contexts, negs) in val_pbar:
                centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
                
                batch_loss = model(centers, contexts, negs, False, 0.0).mean().item()
                total_val_loss += batch_loss
                val_pbar.set_postfix({'val_loss': f'{total_val_loss / (i + 1):.4f}'})
        
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  
            best_model_state = model.state_dict()          
            save_data = {
                'model_state_dict': best_model_state,
                'nodes': nodes,
                'vocab_size': len(nodes),
                'embedding_dim': embedding_dim
            }
            torch.save(save_data, "best_model.pth")        
            print(f"  -> Best model (val_loss={best_val_loss:.4f}), saved to best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save plot
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'o-', label='Train', linewidth=2, markersize=6)
        plt.plot(val_losses, 's-', label='Validation', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150)
        print("\nSaved loss plot to training_loss.png")
        plt.close()
    
    return {
        'nodes': nodes,
        'embeddings': model.get_embeddings(),
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


# ============================================================================
# Analysis
# ============================================================================

def find_similar_words(word, nodes, embeddings, top_k=10):
    """Find most similar words using cosine similarity."""
    if word not in nodes:
        return []
    
    idx = nodes.index(word)
    target_vec = embeddings[idx]
    
    similarities = (embeddings @ target_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
    top_indices = np.argsort(-similarities)[1:top_k+1]
    
    return [(nodes[i], float(similarities[i])) for i in top_indices]


def solve_analogy(word_a, word_b, word_c, nodes, embeddings, top_k=5):
    """Solve word analogies: word_a is to word_b as word_c is to ?"""
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    if not all(w in node_to_idx for w in [word_a, word_b, word_c]):
        return []
    
    target_vec = embeddings[node_to_idx[word_b]] - embeddings[node_to_idx[word_a]] + embeddings[node_to_idx[word_c]]
    similarities = (embeddings @ target_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
    
    exclude = {node_to_idx[w] for w in [word_a, word_b, word_c]}
    results = [(nodes[i], float(similarities[i])) for i in np.argsort(-similarities) if i not in exclude][:top_k]
    
    return results


def visualize_embeddings(nodes, embeddings, output_file="embeddings_tsne.png", 
                        sample_size=200, annotate=True):
    """Create t-SNE visualization of embeddings."""
    n_samples = min(sample_size, len(nodes))
    selected_embeddings = embeddings[:n_samples]
    selected_nodes = nodes[:n_samples]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1)) ## dimentionality reduction for the visualisation
    projection = tsne.fit_transform(selected_embeddings)
    
    plt.figure(figsize=(14, 14))
    plt.scatter(projection[:, 0], projection[:, 1], s=40, alpha=0.6, c='steelblue')
    
    if annotate:
        for i, word in enumerate(selected_nodes):
            plt.annotate(word, (projection[i, 0], projection[i, 1]), fontsize=9, alpha=0.8)
    
    plt.title(f"t-SNE Visualization of Top {n_samples} Word Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE to {output_file}")
    plt.close()


def analyze_embeddings(nodes, embeddings, 
                       similarity_examples=None,
                       analogy_examples=None,
                       cluster_seeds=None):
    """Comprehensive analysis of learned embeddings."""
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    print(f"\nVocabulary: {len(nodes):,}  Embedding dim: {embeddings.shape[1]}")
    
    # Similarity statistics
    sample_emb = embeddings[:min(100, len(embeddings))]
    norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
    normalized = sample_emb / (norms + 1e-10)
    sim_matrix = normalized @ normalized.T
    sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    
    print(f"\nSimilarity stats (100 word sample):")
    print(f"  Mean: {sim_values.mean():.4f}  Std: {sim_values.std():.4f}")
    print(f"  Min: {sim_values.min():.4f}  Max: {sim_values.max():.4f}")
    
    # Nearest neighbors
    if similarity_examples:
        print("\n" + "="*80)
        print("NEAREST NEIGHBORS")
        print("="*80)
        for word in similarity_examples:
            similar = find_similar_words(word, nodes, embeddings, top_k=8)
            print(f"\nMost similar to '{word}':")
            if not similar:
                print("  (not in vocabulary)")
            else:
                for token, score in similar:
                    print(f"  {token:15s}  similarity={score:.4f}")
    
    # Analogies
    if analogy_examples:
        print("\n" + "="*80)
        print("WORD ANALOGIES (a:b :: c:?)")
        print("="*80)
        for a, b, c in analogy_examples:
            results = solve_analogy(a, b, c, nodes, embeddings, top_k=3)
            print(f"\n{a}:{b} :: {c}:?")
            if results:
                for token, score in results:
                    print(f"  {token:15s}  score={score:.4f}")
            else:
                print("  (words not in vocabulary)")
    
    # Semantic clusters
    if cluster_seeds:
        print("\n" + "="*80)
        print("SEMANTIC CLUSTERS")
        print("="*80)
        for seed in cluster_seeds:
            if seed in nodes:
                cluster = find_similar_words(seed, nodes, embeddings, top_k=5)
                print(f"\n'{seed}': {', '.join([w for w, _ in cluster])}")
    
    print("\n" + "="*80)


"""
Unit Tests for Skip-Gram with Negative Sampling

Starter skeleton for testing SkipGramDataset and SkipGramModel classes.
Students should implement their own test cases inside the provided class structures.
"""

class TestSkipGramDataset(unittest.TestCase):
    """Tests for SkipGramDataset class."""
    
    def setUp(self):
        """Create a small test graph."""
        # Create simple graph
        self.graph = nx.Graph()
        self.graph.add_edges_from([
            ("cat", "animal"),
            ("dog", "animal"),
            ("cat", "pet"),
            ("dog", "pet")
        ])
        
        self.nodes = ["cat", "dog", "animal", "pet"]
        
        # Simple distance matrix (all distances = 1)
        self.distance_matrix = np.ones((4, 4))
        np.fill_diagonal(self.distance_matrix, 0)
    

    # IMPLEMENT YOUR TESTS
    def test_build_contexts(self):
        """Test context building."""
        dataset = SkipGramDataset(
            self.graph, 
            self.nodes, 
            self.distance_matrix,
            num_negative=5,
            context_size=1
        )
        
        # Check "cat" has correct contexts
        self.assertIn("animal", dataset.contexts["cat"])
        self.assertIn("pet", dataset.contexts["cat"])
        self.assertEqual(len(dataset.contexts["cat"]), 2)

    def test_pairs_generated(self):
        """Test that pairs are created."""
        dataset = SkipGramDataset(
            self.graph,
            self.nodes,
            self.distance_matrix,
            num_negative=5
        )
        
        # Should have 2*4 = 8 pairs (each edge counted twice)
        self.assertEqual(len(dataset.pairs), 8)
        
        # Weights should have same length
        self.assertEqual(len(dataset.weights), 8)

    def test_getitem(self):
        """Test sampling a training example."""
        dataset = SkipGramDataset(
            self.graph,
            self.nodes,
            self.distance_matrix,
            num_negative=5
        )
        
        center, context, negatives = dataset[0]
        
        # Check types
        self.assertEqual(center.dtype, np.int64)
        self.assertEqual(context.dtype, np.int64)
        self.assertEqual(negatives.dtype, np.int64)
        
        # Check shapes
        self.assertEqual(negatives.shape, (5,))
        
        # Check negatives don't include center
        self.assertNotIn(center, negatives)
    


class TestSkipGramModel(unittest.TestCase):
    """Tests for SkipGramModel class."""
    
    def setUp(self):
        """Create a small model."""
        self.vocab_size = 100
        self.embedding_dim = 16
        self.model = SkipGramModel(self.vocab_size, self.embedding_dim)
    

    def test_initialization(self):
        """Test embeddings are properly initialized."""
        # Check shapes
        self.assertEqual(
            self.model.center_embeddings.weight.shape,
            (self.vocab_size, self.embedding_dim)
        )
        self.assertEqual(
            self.model.context_embeddings.weight.shape,
            (self.vocab_size, self.embedding_dim)
        )
        
        # Check values are in expected range
        center_weights = self.model.center_embeddings.weight.data
        scale = 0.5 / self.embedding_dim
        self.assertTrue(torch.all(center_weights >= -scale))
        self.assertTrue(torch.all(center_weights <= scale))  

    def test_forward_shapes(self):
        """Test forward pass produces correct shapes."""
        batch_size = 4
        num_negatives = 10
        
        center = torch.randint(0, self.vocab_size, (batch_size,))
        context = torch.randint(0, self.vocab_size, (batch_size,))
        negatives = torch.randint(0, self.vocab_size, (batch_size, num_negatives))
        
        loss = self.model(center, context, negatives)
        
        # Loss should be one value per example
        self.assertEqual(loss.shape, (batch_size,))
    
    def test_get_embeddings(self):
        """Test embedding extraction."""
        embeddings = self.model.get_embeddings()
        
        # Check type and shape
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (self.vocab_size, self.embedding_dim))   


class TestIntegration(unittest.TestCase):
    """Integration tests for dataset and model working together."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        graph_raw = nx.karate_club_graph()
        
        self.graph = nx.Graph()
        for u, v in graph_raw.edges():
            self.graph.add_edge(str(u), str(v))
        
        self.nodes = [str(n) for n in sorted(graph_raw.nodes())]
        
        self.dist_matrix = np.ones((len(self.nodes), len(self.nodes)))
        np.fill_diagonal(self.dist_matrix, 0)
        
    def test_end_to_end_training(self):
        """Test that we can train for 1 epoch without errors."""
        # Create dataset
        dataset = SkipGramDataset(
            self.graph, 
            self.nodes, 
            self.dist_matrix,
            num_negative=5,
            context_size=1
        )
        
        # Verify dataset has pairs
        self.assertGreater(len(dataset.pairs), 0, "Dataset should have training pairs!")
        
        model = SkipGramModel(len(self.nodes), embedding_dim=8, dropout=0.1)
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        centers, contexts, negs = next(iter(loader))
        loss = model(centers, contexts, negs).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss should not be Inf")




def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING SKIP-GRAM UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramModel))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("[OK] ALL TESTS PASSED!")
        print(f"Total tests run: {result.testsRun}")
    else:
        print("[X] SOME TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

"""
Skip-Gram Embeddings -- Entry Point
====================================
Loads the trained 523-word Skip-Gram model (best_skipgram_523words.pth) and
exposes build_my_embeddings() for evaluation and downstream use.

Dependencies:  torch  torchvision  numpy  warnings
See requirements.txt for pinned versions.
"""



import torch
import torchvision
import numpy as np
import warnings
from typing import List, Dict, Tuple

from skipgram_trainer import SkipGramModel

# ==========================================================================
# CONSTANTS
# ==========================================================================

DESIRED_WORDS_TASK5 = {
    'a', 'above', 'against', 'air', 'airplane', 'along', 'an', 'and', 'animal', 'apple',
    'aquarium_fish', 'are', 'area', 'arm', 'around', 'at', 'attached', 'baby', 'back',
    'background', 'backpack', 'bag', 'ball', 'banana', 'bananas', 'base', 'baseball',
    'basket', 'bat', 'bathroom', 'beach', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'behind', 'beige', 'being', 'bench', 'beside', 'between', 'bicycle', 'big', 'bike',
    'bird', 'black', 'blanket', 'blonde', 'blue', 'board', 'boat', 'body', 'book',
    'bottle', 'bottom', 'bowl', 'box', 'boy', 'branch', 'branches', 'bread', 'brick',
    'bridge', 'bright', 'broccoli', 'brown', 'building', 'buildings', 'bunch', 'bus',
    'bush', 'bushes', 'butterfly', 'button', 'by', 'cabinet', 'cake', 'camel', 'camera',
    'can', 'cap', 'car', 'carrying', 'cars', 'castle', 'cat', 'caterpillar', 'cattle',
    'ceiling', 'cell', 'cement', 'chain', 'chair', 'cheese', 'child', 'chimpanzee',
    'city', 'clear', 'clock', 'closed', 'cloth', 'cloud', 'clouds', 'cloudy', 'coat',
    'cockroach', 'coffee', 'collar', 'color', 'colored', 'colorful', 'computer',
    'concrete', 'container', 'corner', 'couch', 'counter', 'court', 'cover', 'covered',
    'cow', 'cows', 'crab', 'crocodile', 'cup', 'curtain', 'dark', 'design', 'desk',
    'dinosaur', 'dirt', 'dish', 'display', 'distance', 'dog', 'dolphin', 'donut', 'door',
    'double', 'down', 'dress', 'ear', 'ears', 'eating', 'edge', 'elephant', 'elephants',
    'empty', 'end', 'engine', 'eye', 'eyes', 'face', 'feet', 'fence', 'field', 'fire',
    'flag', 'flatfish', 'floor', 'flower', 'flowers', 'flying', 'food', 'foot', 'for',
    'forest', 'fork', 'four', 'fox', 'frame', 'frisbee', 'from', 'front', 'fruit', 'full',
    'fur', 'game', 'giraffe', 'giraffes', 'girl', 'glass', 'glasses', 'glove', 'gold',
    'grass', 'gray', 'green', 'grey', 'ground', 'group', 'growing', 'guy', 'hair',
    'hamster', 'hand', 'handle', 'hands', 'hanging', 'has', 'hat', 'have', 'he', 'head',
    'helmet', 'her', 'hill', 'his', 'holding', 'horse', 'horses', 'hot', 'house',
    'hydrant', 'in', 'inside', 'into', 'is', 'it', 'jacket', 'jeans', 'jet', 'kangaroo',
    'keyboard', 'kitchen', 'kite', 'knife', 'lady', 'lamp', 'laptop', 'large',
    'lawn_mower', 'laying', 'leaf', 'leaves', 'left', 'leg', 'legs', 'leopard', 'letter',
    'lettering', 'letters', 'license', 'lid', 'light', 'lights', 'line', 'lines', 'lion',
    'little', 'lizard', 'lobster', 'logo', 'long', 'looking', 'lot', 'luggage', 'made',
    'man', "man's", 'many', 'maple_tree', 'men', 'metal', 'middle', 'mirror', 'monitor',
    'motorcycle', 'mountain', 'mountains', 'mouse', 'mouth', 'mushroom', 'napkin',
    'near', 'neck', 'next', 'no', 'nose', 'number', 'numbers', 'oak_tree', 'ocean', 'of',
    'off', 'old', 'on', 'one', 'open', 'orange', 'orchid', 'other', 'otter', 'out',
    'outside', 'oven', 'over', 'painted', 'pair', 'palm_tree', 'pants', 'paper', 'park',
    'parked', 'parking', 'part', 'passenger', 'patch', 'pavement', 'pear', 'people',
    'person', "person's", 'phone', 'photo', 'pickup_truck', 'picture', 'piece', 'pile',
    'pillow', 'pine_tree', 'pink', 'pizza', 'plain', 'plane', 'plant', 'plastic', 'plate',
    'platform', 'player', 'playing', 'pole', 'poles', 'poppy', 'porcupine', 'possum',
    'post', 'pot', 'purple', 'rabbit', 'raccoon', 'rack', 'racket', 'rail', 'railing',
    'ray', 'rear', 'red', 'reflection', 'remote', 'riding', 'right', 'road', 'rock',
    'rocket', 'rocks', 'roof', 'room', 'rose', 'round', 'row', 'sand', 'sandwich',
    'sauce', 'scene', 'screen', 'sea', 'seal', 'seat', 'section', 'set', 'several',
    'shadow', 'shark', 'sheep', 'shelf', 'shirt', 'shoe', 'shoes', 'short', 'shorts',
    'shrew', 'side', 'sidewalk', 'sign', 'signs', 'silver', 'sink', 'sitting',
    'skateboard', 'ski', 'skier', 'skiing', 'skis', 'skunk', 'sky', 'skyscraper',
    'sleeve', 'slice', 'small', 'smiling', 'snail', 'snake', 'snow', 'some', 'spider',
    'spoon', 'spot', 'square', 'squirrel', 'stand', 'standing', 'statue', 'stone', 'stop',
    'stove', 'street', 'streetcar', 'stripe', 'striped', 'stripes', 'stuffed', 'suit',
    'suitcase', 'sun', 'sunflower', 'sunglasses', 'surface', 'surfboard', 'surfer',
    'sweater', 'sweet_pepper', 't', 'table', 'tag', 'tail', 'taken', 'tall', 'tan',
    'tank', 'teddy', 'telephone', 'television', 'tennis', 'that', 'the', 'there', 'these',
    'this', 'three', 'through', 'tie', 'tiger', 'tile', 'tire', 'to', 'toilet', 'top',
    'towel', 'tower', 'track', 'tracks', 'tractor', 'traffic', 'train', 'trash', 'tray',
    'tree', 'trees', 'trout', 'truck', 'trunk', 'tulip', 'turtle', 'tv', 'two',
    'umbrella', 'under', 'up', 'van', 'vase', 'vehicle', 'very', 'view', 'visible',
    'walking', 'wall', 'wardrobe', 'watch', 'watching', 'water', 'wave', 'waves',
    'wearing', 'wears', 'wet', 'whale', 'wheel', 'wheels', 'white', 'willow_tree', 'window',
    'windows', 'windshield', 'wine', 'wing', 'wire', 'with', 'wolf', 'woman', "woman's",
    'wood', 'wooden', 'word', 'worm', 'worn', 'wrist', 'writing', 'yellow', 'young',
    'zebra', 'zebras'
}


TASK_5_WORDS_MAPPING = {
        "aquariumfish": "aquarium_fish",
        "lawnmower": "lawn_mower",
        "pickuptruck": "pickup_truck",
        'maple': 'maple_tree',
        'oak': 'oak_tree',
        'palm': 'palm_tree',
        'pine': 'pine_tree',
        'willow': 'willow_tree',
        'orchids': 'orchid',
        'poppies': 'poppy',
        'roses': 'rose',
        'sunflowers': 'sunflower',
        'tulips': 'tulip',
        'mushrooms': 'mushroom',
        'pears': 'pear',
        'sweetpeppers': 'sweet_pepper',
}


# ============================================================================
# UTILITIES AND HELPERS
# ============================================================================

def load_embeddings_core(
    checkpoint_path: str,
    load_from_model: bool = False,
    verbose: bool = False
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load Skip-gram embeddings from checkpoint (shared by Tasks 5 & 6).
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        load_from_model: If True, instantiate SkipGramModel and extract embeddings
        verbose: Print loading details
    
    Returns:
        vocab: Word-to-index mapping for all words in checkpoint
        embeddings: Embedding matrix (vocab_size, embedding_dim)
    
    """

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            f"Please run final_check.py first to generate this file."
        )
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")
    
    # Extract vocabulary list
    if 'nodes' not in checkpoint:
        raise KeyError("Checkpoint must contain 'nodes' (vocabulary list).")
    
    vocab_list = checkpoint['nodes']
    
    # Get embeddings
    if load_from_model:
        # Load from model
        if 'vocab_size' not in checkpoint or 'embedding_dim' not in checkpoint:
            raise KeyError("Checkpoint must contain 'vocab_size' and 'embedding_dim'.")
        
        vocab_size = checkpoint['vocab_size']
        embedding_dim = checkpoint['embedding_dim']
        
        model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=0.35).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with torch.no_grad():
            embeddings_tensor = model.get_embeddings()
            
            if isinstance(embeddings_tensor, torch.Tensor):
                if embeddings_tensor.device.type != 'cpu':
                    embeddings_tensor = embeddings_tensor.cpu()
                embeddings = embeddings_tensor.numpy().astype(np.float32)
            else:
                embeddings = np.array(embeddings_tensor, dtype=np.float32)
    else:
        # Load directly from checkpoint
        if 'embeddings' not in checkpoint:
            raise KeyError("Checkpoint must contain 'embeddings' (embedding matrix).")
        
        embeddings = checkpoint['embeddings']
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy().astype(np.float32)
        else:
            embeddings = np.array(embeddings, dtype=np.float32)
    
    # Create word-to-index mapping
    vocab = {word: idx for idx, word in enumerate(vocab_list)}

    # Validation
    if len(vocab) != embeddings.shape[0]:
        raise ValueError(
            f"Vocabulary size ({len(vocab)}) doesn't match embedding rows ({embeddings.shape[0]})"
        )
    
    if verbose:
        print(f"Embeddings loaded successfully")
        print(f"Vocabulary size: {len(vocab):,} words")
        print(f"Embedding dimension: {embeddings.shape[1]}")
    
    return vocab, embeddings


# ============================================================================
# SECTION 1: CIFAR-100 SEMANTIC EXPANSION
# ============================================================================

# DO NOT CHANGE THIS FUNCTION's signature
def build_my_embeddings(checkpoint_path: str = "best_skipgram_523words.pth") -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load and return your trained Skip-gram embeddings.
    
    This function serves as the entry point for loading your final embedding model
    that contains all Visual Genome words AND all 100 CIFAR-100 classes.
    
    Args:
        checkpoint_path: Path to your saved model checkpoint
        
    Returns:
        vocab: Dictionary mapping words to indices {word: index}
        embeddings: Numpy array of shape (vocab_size, embedding_dim)
        
    Example:
        >>> vocab, embeddings = build_my_embeddings()
        >>> print(f"Vocabulary size: {len(vocab)}")
        >>> print(f"Embedding dimension: {embeddings.shape[1]}")
        >>> print(f"'airplane' index: {vocab.get('airplane', 'NOT FOUND')}")
    """
    VERBOSE=False
    vocab, embeddings = load_embeddings_core(
        checkpoint_path,
        load_from_model=False,
        verbose=VERBOSE
    )
    
    # Handle words mapping for task 5
    for missing_word, replacement in TASK_5_WORDS_MAPPING.items():
        if missing_word in vocab:
            if VERBOSE:
                print(f"index of {missing_word}: {vocab[missing_word]}")
            vocab[replacement] = vocab.pop(missing_word)

    # get required 523 words
    filtered_vocab = {}
    filtered_embs = []

    for word in DESIRED_WORDS_TASK5:
        if word in vocab:
            filtered_vocab[word] = len(filtered_vocab)
            filtered_embs.append(embeddings[vocab[word]])

    filtered_embs = np.array(filtered_embs).astype(np.float32)

    ## Normalize embeddings
    eps = 1e-12
    norms = np.linalg.norm(filtered_embs, axis=1, keepdims=True)
    filtered_embs = filtered_embs / np.maximum(norms, eps)

    if VERBOSE:
        print(f"Filtered vocab size: {len(filtered_vocab)}")
        print(f"Filtered embedding shape: {filtered_embs.shape}")

    # Verify CIFAR-100 words are present
    _ds = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True)
    cifar_vocab = _ds.classes

    missing_cifar = [w for w in cifar_vocab if w not in filtered_vocab]
    if missing_cifar:
        warnings.warn(
            f"Some CIFAR-100 words missing from vocabulary: {missing_cifar}\n"
        )
    
    if filtered_embs.shape[0] < 523:
        warnings.warn(
            f"Vocabulary size ({filtered_embs.shape[0]}) is smaller than required minimum (523 words)"
        )
    
    if VERBOSE:
        print(f"  Embeddings loaded successfully")
        print(f"  Vocabulary size: {len(filtered_vocab):,} words")
        print(f"  Embedding dimension: {filtered_embs.shape[1]}")
    
    return (filtered_vocab, filtered_embs)


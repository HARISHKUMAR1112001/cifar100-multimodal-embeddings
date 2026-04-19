import numpy as np
from collections import defaultdict
import torch
from skipgram_embeddings import build_my_embeddings, DESIRED_WORDS_TASK5 as DESIRED_WORDS
from skipgram_trainer import find_similar_words, analyze_embeddings, visualize_embeddings, SkipGramModel
import os

vocab , embeddings = build_my_embeddings()
loaded_vocab = sorted(vocab.keys(), key=lambda w: vocab[w])
loaded_embeddings = embeddings[[vocab[w] for w in loaded_vocab]]
    
# ===========================================================================
# COMPLETE CIFAR-100 DATASET (from official documentation)
# ===========================================================================

SUPERCLASSES = {
    # Row 1
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    
    # Row 2
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    
    # Row 3
    'fruits_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    
    # Row 4
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    
    # Row 5
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man_made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    
    # Row 6
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    
    # Row 7
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non_insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    
    # Row 8
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    
    # Row 9
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    
    # Row 10
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

# ===========================================================================
# EXTRACT ALL CIFAR-100 WORDS
# ===========================================================================

ALL_CIFAR100_WORDS = []
for superclass, members in SUPERCLASSES.items():
    ALL_CIFAR100_WORDS.extend(members)

print(f"\n Total CIFAR-100 words: {len(ALL_CIFAR100_WORDS)}")
print(f" Total superclasses: {len(SUPERCLASSES)}")

# ===========================================================================
# BIDIRECTIONAL SEMANTIC ALLOWANCES
# ===========================================================================

SEMANTIC_ALLOWANCES = {
    # Natural scenes <-> Trees
    'forest': {'allowed_classes': ['trees'], 'reason': 'Trees are IN forests'},
    'mountain': {'allowed_classes': ['trees'], 'reason': 'Trees grow on mountains'},
    'plain': {'allowed_classes': ['trees'], 'reason': 'Trees grow on plains'},
    'sea': {'allowed_classes': ['aquatic_mammals', 'fish', 'trees'], 'reason': 'Marine life + coastal trees'},
    'cloud': {'allowed_classes': ['trees'], 'reason': 'Clouds over forests'},
    
    # Trees <-> Natural scenes (reverse)
    'maple_tree': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Trees grow IN landscapes'},
    'oak_tree': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Trees grow IN landscapes'},
    'willow_tree': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Willows near water'},
    'palm_tree': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Palms on beaches/plains'},
    'pine_tree': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Pines in forests/mountains'},
    
    # Aquatic mammals <-> Sea
    'beaver': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Beavers live near water'},
    'dolphin': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Dolphins in seas'},
    'otter': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Otters in seas/rivers'},
    'seal': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Seals in seas'},
    'whale': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Whales in seas'},
    
    # Invertebrates in nature
    'crab': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Crabs near seas'},
    'lobster': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Lobsters in seas'},
    'spider': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Spiders in forests'},
    'worm': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Worms in soil/forests'},
    'snail': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Snails in forests'},
    
    # People <-> Man-made structures
    'baby': {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'boy': {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'girl': {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'man': {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'woman': {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    
    # Vehicles on roads/infrastructure
    'bicycle': {'allowed_classes': ['large_man_made_outdoor_things'], 'reason': 'Bicycles on roads/bridges'},
    'bus': {'allowed_classes': ['large_man_made_outdoor_things'], 'reason': 'Buses on roads'},
    'motorcycle': {'allowed_classes': ['large_man_made_outdoor_things'], 'reason': 'Motorcycles on roads'},
    'pickup_truck': {'allowed_classes': ['large_man_made_outdoor_things', 'vehicles_2'], 'reason': 'Trucks on roads, semantically a vehicle'},
    'train': {'allowed_classes': ['large_man_made_outdoor_things'], 'reason': 'Trains on infrastructure'},
    
    # Carnivores in nature
    'bear': {'allowed_classes': ['large_natural_outdoor_scenes', 'trees'], 'reason': 'Bears in forests'},
    'leopard': {'allowed_classes': ['large_natural_outdoor_scenes', 'trees'], 'reason': 'Leopards in wild'},
    'lion': {'allowed_classes': ['large_natural_outdoor_scenes'], 'reason': 'Lions in plains'},
    'tiger': {'allowed_classes': ['large_natural_outdoor_scenes', 'trees'], 'reason': 'Tigers in jungles'},
    'wolf': {'allowed_classes': ['large_natural_outdoor_scenes', 'trees'], 'reason': 'Wolves in forests'},
}

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def get_word_to_superclass():
    """Create mapping from word to its superclass."""
    word_to_class = {}
    for superclass, members in SUPERCLASSES.items():
        for word in members:
            word_to_class[word] = superclass
    return word_to_class

def is_semantically_valid_neighbor(word, neighbor_word, neighbor_class):
    """Check if a cross-class neighbor is semantically valid."""
    if word not in SEMANTIC_ALLOWANCES:
        return False, None
    
    allowance = SEMANTIC_ALLOWANCES[word]
    if neighbor_class in allowance['allowed_classes']:
        return True, allowance['reason']
    
    return False, None

def compute_intra_class_similarity(superclass, members, vocab, embeddings):
    """Compute average similarity within a superclass."""
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    
    valid_members = [w for w in members if w in word_to_idx]
    if len(valid_members) < 2:
        return None, None, valid_members
    
    member_indices = [word_to_idx[w] for w in valid_members]
    member_vecs = embeddings[member_indices]
    
    norms = np.linalg.norm(member_vecs, axis=1, keepdims=True) + 1e-10
    member_vecs_norm = member_vecs / norms
    
    sim_matrix = member_vecs_norm @ member_vecs_norm.T
    
    n = len(valid_members)
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            similarities.append(sim_matrix[i, j])
    
    if not similarities:
        return None, None, valid_members
    
    return np.mean(similarities), np.std(similarities), valid_members

# ===========================================================================
# TEST 1: ALL CIFAR-100 WORD EMBEDDINGS
# ===========================================================================

def test_all_cifar100_words():
    """Test individual word embeddings and nearest neighbors."""
    print("\n" + "="*80)
    print("TEST 1: ALL CIFAR-100 WORD EMBEDDINGS")
    print("="*80)
    
    word_to_class = get_word_to_superclass()
    missing = []
    present = []
    
    for i, word in enumerate(ALL_CIFAR100_WORDS, 1):
        if word not in loaded_vocab:
            missing.append(word)
            print(f"[{i:3d}/100] [X] '{word:20s}' -> NOT IN VOCABULARY")
            continue
        
        present.append(word)
        idx = loaded_vocab.index(word)
        norm = np.linalg.norm(loaded_embeddings[idx])
        superclass = word_to_class.get(word, "unknown")
        
        similar = find_similar_words(word, loaded_vocab, loaded_embeddings, top_k=8)
        
        if not similar:
            print(f"[{i:3d}/100] [!]  '{word:20s}' -> idx={idx:4d}, norm={norm:.4f}, class={superclass}")
            continue
        
        neighbor_str = ', '.join([f"{w}({s:.3f})" for w, s in similar])
        
        print(f"[{i:3d}/100] [OK] '{word:20s}' -> norm={norm:.4f}, class={superclass}")
        print(f"          Neighbors: {neighbor_str}")
    
    print("\n" + "="*80)
    print(f"[OK] Present in vocabulary: {len(present)}/100")
    if missing:
        print(f"[X] Missing from vocabulary: {len(missing)}/100")
        print(f"   Missing words: {', '.join(missing)}")
    else:
        print(" ALL 100 CIFAR-100 WORDS PRESENT IN VOCABULARY!")
    print("="*80)

# ===========================================================================
# TEST 2: SUPERCLASS CLUSTERING VALIDATION
# ===========================================================================

def test_superclass_clustering():
    """Validate that words in same superclass cluster together."""
    print("\n" + "="*80)
    print("TEST 2: SUPERCLASS CLUSTERING VALIDATION (ALL 20 CLASSES)")
    print("="*80)
    
    results = []
    
    for superclass, members in SUPERCLASSES.items():
        mean_sim, std_sim, valid_members = compute_intra_class_similarity(
            superclass, members, loaded_vocab, loaded_embeddings
        )
        
        if mean_sim is None:
            print(f"\n[!]  {superclass:35s} -> Not enough members in vocab")
            continue
        
        if mean_sim >= 0.80:
            status = "[OK] EXCELLENT"
        elif mean_sim >= 0.70:
            status = "[OK] GOOD"
        elif mean_sim >= 0.60:
            status = "[!]  FAIR"
        else:
            status = "[X] POOR"
        
        results.append((superclass, mean_sim, std_sim, len(valid_members)))
        
        print(f"\n{status}  {superclass:35s}")
        print(f"   Members ({len(valid_members)}/5): {', '.join(valid_members)}")
        print(f"   Avg similarity: {mean_sim:.4f} +/- {std_sim:.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY: SUPERCLASS COHESION (20 CLASSES)")
    print("="*80)
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    excellent = sum(1 for _, sim, _, _ in results if sim >= 0.80)
    good = sum(1 for _, sim, _, _ in results if 0.70 <= sim < 0.80)
    fair = sum(1 for _, sim, _, _ in results if 0.60 <= sim < 0.70)
    poor = sum(1 for _, sim, _, _ in results if sim < 0.60)
    
    print(f"[OK] Excellent (>=0.80): {excellent}/20")
    print(f"[OK] Good (0.70-0.80):  {good}/20")
    print(f"[!]  Fair (0.60-0.70):  {fair}/20")
    print(f"[X] Poor (<0.60):      {poor}/20")
    
    print(f"\nTop 5 Best Clustering:")
    for superclass, mean_sim, std_sim, n_members in results[:5]:
        print(f"  {superclass:35s}  sim={mean_sim:.4f} ({n_members} words)")
    
    if poor > 0:
        print(f"\nBottom 5 Worst Clustering:")
        for superclass, mean_sim, std_sim, n_members in results[-5:]:
            print(f"  {superclass:35s}  sim={mean_sim:.4f} ({n_members} words)")

# ===========================================================================
# TEST 3: CROSS-CLASS CONTAMINATION (SEMANTIC-AWARE)
# ===========================================================================

def test_cross_class_contamination_semantic_aware():
    """Check for INVALID cross-class contamination with semantic awareness."""
    print("\n" + "="*80)
    print("TEST 3: CROSS-CLASS CONTAMINATION CHECK (SEMANTIC-AWARE)")
    print("="*80)
    
    word_to_class = get_word_to_superclass()
    invalid_contamination = []
    valid_semantic_neighbors = []
    
    for word in ALL_CIFAR100_WORDS:
        if word not in loaded_vocab:
            continue
        
        true_class = word_to_class.get(word, "unknown")
        similar = find_similar_words(word, loaded_vocab, loaded_embeddings, top_k=10)
        
        same_class_count = 0
        invalid_neighbors = []
        semantic_neighbors = []
        
        for neighbor_word, sim in similar[:5]:
            if neighbor_word in word_to_class:
                neighbor_class = word_to_class[neighbor_word]
                
                if neighbor_class == true_class:
                    same_class_count += 1
                else:
                    is_valid, reason = is_semantically_valid_neighbor(word, neighbor_word, neighbor_class)
                    
                    if is_valid:
                        semantic_neighbors.append((neighbor_word, neighbor_class, sim, reason))
                    else:
                        invalid_neighbors.append((neighbor_word, neighbor_class, sim))
        
        if semantic_neighbors:
            valid_semantic_neighbors.append((word, true_class, semantic_neighbors))
        
        if same_class_count < 3 and invalid_neighbors:
            invalid_contamination.append((word, true_class, same_class_count, invalid_neighbors))
    
    # Report valid semantic neighbors
    if valid_semantic_neighbors:
        print(f"\n[OK] {len(valid_semantic_neighbors)} words with VALID semantic cross-class neighbors:\n")
        for word, true_class, neighbors in valid_semantic_neighbors[:10]:
            print(f"[OK] '{word}' (class: {true_class})")
            for neighbor, neighbor_class, sim, reason in neighbors[:3]:
                print(f"   -> {neighbor}[{neighbor_class}]({sim:.3f}) - {reason}")
        
        # if len(valid_semantic_neighbors) > 10:
        #     print(f"\n   ... and {len(valid_semantic_neighbors) - 10} more with valid semantic relationships")
    
    # Report invalid contamination
    print("\n" + "="*80)
    if invalid_contamination:
        print(f"[!]  Found {len(invalid_contamination)} words with TRULY INVALID contamination:\n")
        for word, true_class, same_count, others in invalid_contamination[:20]:
            print(f"[X] '{word}' (class: {true_class})")
            print(f"   Same-class in top-5: {same_count}/5")
            print(f"   Invalid neighbors: {', '.join([f'{w}[{c}]({s:.2f})' for w, c, s in others[:3]])}")
        
        if len(invalid_contamination) > 1000: ##20
            print(f"\n   ... and {len(invalid_contamination) - 20} more contaminated words")
    else:
        print(" ZERO INVALID CROSS-CLASS CONTAMINATION!")
        print("   All cross-class neighbors are semantically valid!")
    
    print("\n" + "="*80)
    print(" FINAL CONTAMINATION SUMMARY")
    print("="*80)
    print(f"[OK] Semantically valid relationships: {len(valid_semantic_neighbors)}/100")
    print(f"[X] Truly invalid contamination: {len(invalid_contamination)}/100")
    print(f" Effective contamination rate: {len(invalid_contamination)/100*100:.1f}%")
    
    perfect_words = 100 - len(valid_semantic_neighbors) - len(invalid_contamination)
    print(f"\n QUALITY BREAKDOWN:")
    print(f"   * Perfect clustering: {perfect_words}/100 ({perfect_words/100*100:.1f}%)")
    print(f"   [OK] Semantic relationships: {len(valid_semantic_neighbors)}/100 ({len(valid_semantic_neighbors)/100*100:.1f}%)")
    print(f"   [X] True contamination: {len(invalid_contamination)}/100 ({len(invalid_contamination)/100*100:.1f}%)")
    print("="*80)



# ===========================================================================
# TEST 4: EVALUATE ALL DESIRED_WORDS (423 words)
# ===========================================================================

# Semantic categories for DESIRED_WORDS (beyond CIFAR-100)
WORD_CATEGORIES = {
    'colors': {'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'pink', 
               'purple', 'brown', 'gray', 'grey', 'gold', 'silver', 'beige', 'tan', 'blonde'},
    
    'body_parts': {'arm', 'hand', 'hands', 'leg', 'legs', 'foot', 'feet', 'head', 'face',
                   'eye', 'eyes', 'ear', 'ears', 'nose', 'mouth', 'hair', 'neck', 'wrist', 'tail'},
    
    'clothing': {'shirt', 'pants', 'jeans', 'dress', 'coat', 'jacket', 'sweater', 'hat',
                 'cap', 'helmet', 'shoe', 'shoes', 'shorts', 'suit', 'tie', 'glove', 
                 'sunglasses', 'glasses', 'sleeve'},
    
    'furniture': {'bed', 'chair', 'couch', 'table', 'desk', 'bench', 'shelf', 'cabinet',
                  'wardrobe', 'counter', 'rack', 'stand'},
    
    'food': {'apple', 'banana', 'bananas', 'orange', 'bread', 'cake', 'pizza', 'donut',
             'sandwich', 'broccoli', 'cheese', 'coffee', 'wine', 'sauce', 'fruit'},
    
    'kitchen_items': {'bowl', 'plate', 'cup', 'bottle', 'fork', 'knife', 'spoon', 'pot',
                      'dish', 'container', 'tray', 'napkin', 'lid'},
    
    'vehicles': {'car', 'cars', 'bus', 'truck', 'train', 'boat', 'airplane', 'plane',
                 'bike', 'bicycle', 'motorcycle', 'van', 'jet'},
    
    'nature': {'tree', 'trees', 'flower', 'flowers', 'grass', 'leaf', 'leaves', 'branch',
               'branches', 'bush', 'bushes', 'plant', 'rock', 'rocks', 'sand', 'dirt',
               'water', 'wave', 'waves', 'ocean', 'sea', 'sky', 'cloud', 'clouds', 'sun',
               'snow', 'forest', 'mountain', 'mountains', 'hill', 'field'},
    
    'buildings': {'house', 'building', 'buildings', 'tower', 'bridge', 'castle', 'room',
                  'kitchen', 'bathroom', 'window', 'windows', 'door', 'wall', 'ceiling',
                  'floor', 'roof'},
    
    'animals': {'dog', 'cat', 'bird', 'horse', 'horses', 'cow', 'cows', 'sheep', 'bear',
                'elephant', 'elephants', 'giraffe', 'giraffes', 'zebra', 'zebras', 'lion',
                'tiger', 'monkey'},
    
    'sports': {'ball', 'baseball', 'tennis', 'surfboard', 'skateboard', 'ski', 'skis',
               'racket', 'frisbee', 'player', 'game', 'court'},
    
    'electronics': {'phone', 'computer', 'laptop', 'keyboard', 'monitor', 'screen', 'tv',
                    'television', 'camera', 'remote', 'lamp', 'clock', 'watch'},
    
    'positions': {'top', 'bottom', 'left', 'right', 'front', 'back', 'side', 'middle',
                  'corner', 'edge', 'end', 'center'},
    
    'prepositions': {'in', 'on', 'at', 'to', 'from', 'with', 'by', 'for', 'of', 'above',
                     'below', 'under', 'over', 'near', 'next', 'behind', 'between',
                     'through', 'into', 'out', 'up', 'down', 'along', 'around', 'against'},
    
    'actions': {'standing', 'sitting', 'walking', 'riding', 'playing', 'holding', 'wearing',
                'looking', 'eating', 'flying', 'hanging', 'laying', 'carrying', 'watching',
                'smiling', 'growing', 'skiing', 'parked'},
    
    'sizes': {'big', 'small', 'large', 'little', 'tall', 'short', 'long', 'old', 'young'},
    
    'common_objects': {'bag', 'backpack', 'umbrella', 'book', 'box', 'sign', 'signs',
                       'photo', 'picture', 'mirror', 'towel', 'pillow', 'blanket'},
}

def test_all_desired_words():
    """Test all 423 DESIRED_WORDS for vocabulary coverage."""
    print("\n" + "="*80)
    print("TEST 4: ALL DESIRED_WORDS VOCABULARY COVERAGE")
    print("="*80)
    
    present = []
    missing = []
    
    for word in sorted(DESIRED_WORDS):
        if word in loaded_vocab:
            present.append(word)
        else:
            missing.append(word)
    
    print(f"\n VOCABULARY COVERAGE:")
    print(f"   [OK] Present: {len(present)}/{len(DESIRED_WORDS)} ({len(present)/len(DESIRED_WORDS)*100:.1f}%)")
    print(f"   [X] Missing: {len(missing)}/{len(DESIRED_WORDS)} ({len(missing)/len(DESIRED_WORDS)*100:.1f}%)")
    
    if missing:
        print(f"\n[X] Missing words ({len(missing)}):")
        for i in range(0, len(missing), 10):
            print(f"   {', '.join(missing[i:i+10])}")
    
    return present, missing


def test_category_clustering():
    """Test clustering within semantic categories."""
    print("\n" + "="*80)
    print("TEST 5: SEMANTIC CATEGORY CLUSTERING")
    print("="*80)
    
    results = []
    
    for category, words in WORD_CATEGORIES.items():
        # Filter to words in vocabulary
        valid_words = [w for w in words if w in loaded_vocab]
        
        if len(valid_words) < 2:
            print(f"\n[!]  {category:20s} -> Not enough words in vocab ({len(valid_words)}/{len(words)})")
            continue
        
        # Compute intra-category similarity
        word_to_idx = {w: i for i, w in enumerate(loaded_vocab)}
        indices = [word_to_idx[w] for w in valid_words]
        vecs = loaded_embeddings[indices]
        
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        vecs_norm = vecs / norms
        
        # Compute pairwise similarities
        sim_matrix = vecs_norm @ vecs_norm.T
        
        similarities = []
        n = len(valid_words)
        for i in range(n):
            for j in range(i+1, n):
                similarities.append(sim_matrix[i, j])
        
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Status
        if mean_sim >= 0.75:
            status = "[OK] EXCELLENT"
        elif mean_sim >= 0.60:
            status = "[OK] GOOD"
        elif mean_sim >= 0.45:
            status = "[!]  FAIR"
        else:
            status = "[X] POOR"
        
        results.append((category, mean_sim, std_sim, len(valid_words), len(words)))
        
        print(f"\n{status}  {category:20s} ({len(valid_words)}/{len(words)} words)")
        print(f"   Avg similarity: {mean_sim:.4f} +/- {std_sim:.4f}")
        print(f"   Words: {', '.join(valid_words[:8])}{'...' if len(valid_words) > 8 else ''}")
    
    # Summary
    print("\n" + "="*80)
    print("CATEGORY CLUSTERING SUMMARY")
    print("="*80)
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    excellent = sum(1 for _, sim, _, _, _ in results if sim >= 0.75)
    good = sum(1 for _, sim, _, _, _ in results if 0.60 <= sim < 0.75)
    fair = sum(1 for _, sim, _, _, _ in results if 0.45 <= sim < 0.60)
    poor = sum(1 for _, sim, _, _, _ in results if sim < 0.45)
    
    print(f"[OK] Excellent (>=0.75): {excellent}/{len(results)}")
    print(f"[OK] Good (0.60-0.75):  {good}/{len(results)}")
    print(f"[!]  Fair (0.45-0.60):  {fair}/{len(results)}")
    print(f"[X] Poor (<0.45):      {poor}/{len(results)}")
    
    return results


def test_word_embedding_quality():
    """Test embedding quality metrics for all DESIRED_WORDS."""
    print("\n" + "="*80)
    print("TEST 6: EMBEDDING QUALITY METRICS")
    print("="*80)
    
    word_to_idx = {w: i for i, w in enumerate(loaded_vocab)}
    
    # Metrics
    norms = []
    neighbor_qualities = []
    
    for word in DESIRED_WORDS:
        if word not in loaded_vocab:
            continue
        
        idx = word_to_idx[word]
        vec = loaded_embeddings[idx]
        norm = np.linalg.norm(vec)
        norms.append((word, norm))
        
        # Check neighbor quality
        similar = find_similar_words(word, loaded_vocab, loaded_embeddings, top_k=5)
        if similar:
            avg_sim = np.mean([s for _, s in similar])
            neighbor_qualities.append((word, avg_sim))
    
    # Norm statistics
    norm_values = [n for _, n in norms]
    print(f"\n EMBEDDING NORMS:")
    print(f"   Mean: {np.mean(norm_values):.4f}")
    print(f"   Std:  {np.std(norm_values):.4f}")
    print(f"   Min:  {np.min(norm_values):.4f} ({min(norms, key=lambda x: x[1])[0]})")
    print(f"   Max:  {np.max(norm_values):.4f} ({max(norms, key=lambda x: x[1])[0]})")
    
    # Neighbor quality
    qual_values = [q for _, q in neighbor_qualities]
    print(f"\n NEIGHBOR SIMILARITY (avg of top-5):")
    print(f"   Mean: {np.mean(qual_values):.4f}")
    print(f"   Std:  {np.std(qual_values):.4f}")
    print(f"   Min:  {np.min(qual_values):.4f}")
    print(f"   Max:  {np.max(qual_values):.4f}")
    
    # Words with best/worst neighbors
    neighbor_qualities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n Top 10 words with best neighbors:")
    for word, sim in neighbor_qualities[:10]:
        neighbors = find_similar_words(word, loaded_vocab, loaded_embeddings, top_k=3)
        neighbor_str = ', '.join([f"{w}({s:.2f})" for w, s in neighbors])
        print(f"   {word:15s} -> {neighbor_str}")
    
    print(f"\n[!]  Bottom 10 words with worst neighbors:")
    for word, sim in neighbor_qualities[-10:]:
        neighbors = find_similar_words(word, loaded_vocab, loaded_embeddings, top_k=3)
        neighbor_str = ', '.join([f"{w}({s:.2f})" for w, s in neighbors])
        print(f"   {word:15s} -> {neighbor_str}")


def test_hub_words():
    """Identify hub words that appear as neighbors too frequently."""
    print("\n" + "="*80)
    print("TEST 7: HUB WORD DETECTION")
    print("="*80)
    
    neighbor_counts = defaultdict(int)
    
    for word in DESIRED_WORDS:
        if word not in loaded_vocab:
            continue
        
        similar = find_similar_words(word, loaded_vocab, loaded_embeddings, top_k=10)
        for neighbor, _ in similar:
            if neighbor in DESIRED_WORDS:
                neighbor_counts[neighbor] += 1
    
    # Sort by frequency
    sorted_counts = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n TOP 20 HUB WORDS (appear most as neighbors):")
    for word, count in sorted_counts[:20]:
        category = None
        for cat, words in WORD_CATEGORIES.items():
            if word in words:
                category = cat
                break
        cat_str = f"[{category}]" if category else ""
        print(f"   {word:15s} {cat_str:20s} appears {count:3d} times")
    
    # Detect potential hub problems
    mean_count = np.mean([c for _, c in sorted_counts])
    std_count = np.std([c for _, c in sorted_counts])
    threshold = mean_count + 2 * std_count
    
    hubs = [(w, c) for w, c in sorted_counts if c > threshold]
    
    print(f"\n HUB STATISTICS:")
    print(f"   Mean appearances: {mean_count:.1f}")
    print(f"   Std: {std_count:.1f}")
    print(f"   Hub threshold (mean+2sigma): {threshold:.1f}")
    print(f"   [!]  Potential hubs (>{threshold:.0f}): {len(hubs)}")
    
    if hubs:
        print(f"\n[!]  POTENTIAL HUB WORDS:")
        for word, count in hubs:
            print(f"   {word}: {count} appearances")

# ===========================================================================
# RUN ALL TESTS
# ===========================================================================

if __name__ == "__main__":
    test_all_cifar100_words()
    test_superclass_clustering()
    test_cross_class_contamination_semantic_aware()

        
    # NEW TESTS FOR ALL DESIRED_WORDS
    test_all_desired_words()
    test_category_clustering()
    test_word_embedding_quality()
    test_hub_words()

    print(analyze_embeddings(loaded_vocab, loaded_embeddings))
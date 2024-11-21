
"""
Quantized Alhabima Panalogy
The quantized panalogy extracts alphabets with specified colors remapped to an shared index
Often we will 'anchor' the alphabet to a background (which may be a variable color)
Concerning the question of how to generalize new test colors?
Elementary pairs globalization perhaps?
"""

from collections import defaultdict
import numpy as np
import string
import os

def abstract_array(quantization_hash, input_array, tile_size, stride_length, starter_alphabet=None, modify_alphabet=True):
    
    alphabet = starter_alphabet

    def extract_tiles(array, tile_size, stride_length):
        tile_depth, tile_rows, tile_cols = tile_size
        depth, rows, cols = array.shape
        tile_depth = min(tile_depth, depth)
        stride_depth, stride_rows, stride_cols = stride_length, stride_length, stride_length
        
        tiles = {}
        dequantizers = {}
        for k in range(0, depth - tile_depth + 1, stride_depth):
            for i in range(0, rows - tile_rows + 1, stride_rows):
                for j in range(0, cols - tile_cols + 1, stride_cols):
                    # We need to copy it so quantization modifications do not propagate
                    tile = array[k:k+tile_depth, i:i+tile_rows, j:j+tile_cols].copy()
                    # print(tile)
                    # Remap colors to quantized indices
                    local_quantizations = {}
                    local_dequantizer = {}
                    for d in range(tile.shape[0]):
                        for y in range(tile.shape[1]):
                            for x in range(tile.shape[2]):
                                pixel_val = tile[d][y][x]
                                if pixel_val in quantization_hash:
                                    # Use local palette quantization
                                    if pixel_val not in local_quantizations:
                                        # print(pixel_val)
                                        # print(len(local_quantizations))
                                        # print(f"Local quant is {10 + len(local_quantizations)}")
                                        palette_index = 10 + len(local_quantizations)
                                        local_quantizations[pixel_val] = palette_index 
                                        local_dequantizer[palette_index] = pixel_val
                                    tile[d][y][x] = local_quantizations[pixel_val]

                    tiles[(k, i, j)] = tile
                    dequantizers[(k, i, j)] = local_dequantizer
        return dequantizers, tiles
    
    def create_alphabet(dequantizers, tiles, starter_alphabet=None):
        alphabet = starter_alphabet.copy() if starter_alphabet else {}
        greek_alphabet_lower = [chr(i) for i in range(0x03B1, 0x03C0)]  # α to ω
        greek_alphabet_upper = [chr(i) for i in range(0x0391, 0x03A0)]  # Α to Ω
        latin_lowercase = [chr(i) for i in range(0x0061, 0x007B)]  # a to z
        latin_uppercase = [chr(i) for i in range(0x0041, 0x005B)]  # A to Z
        japanese_hiragana = [chr(i) for i in range(0x3042, 0x3094)]  # あ to ん
        chinese_characters = [chr(i) for i in range(0x53BB, 0x53BB + 64*64)]  # A set of 64x64 Chinese characters
        
        letters = greek_alphabet_lower + greek_alphabet_upper + latin_lowercase + latin_uppercase + japanese_hiragana + chinese_characters
        
        for (d, i, j), tile in tiles.items():
            tile_key = f"{tile.shape}|{tile.tobytes()}"
            if tile_key not in alphabet:
                alphabet[tile_key] = {'letter': letters[len(alphabet)], 'shape': tile.shape, 'tile': tile, 'deq':dequantizers[(d, i, j)]}
        
        return alphabet
    
    # TODO: Special case for io tiles with zero i->o delta
    def create_abstracted_array(array, tiles, alphabet):
        depth, rows, cols = array.shape
        tile_depth, tile_rows, tile_cols = tile_size
        tile_depth = min(tile_depth, depth)
        abstracted_array = np.empty((depth - tile_depth + 1, rows - tile_rows + 1, cols - tile_cols + 1), dtype=str)
        
        for (d, i, j), tile in tiles.items():
            tile_key = f"{tile.shape}|{tile.tobytes()}"
            abstracted_array[d, i, j] = alphabet[tile_key]['letter']
        
        return abstracted_array
    
    tile_depth, tile_rows, tile_cols = tile_size
    padded_array = np.pad(
        input_array, 
        ((0, 0), (tile_rows-1, tile_rows-1), (tile_cols-1, tile_cols-1)), 
        mode='constant', 
        constant_values=-1
    )
    print(padded_array)
    dequantizers, tiles = extract_tiles(padded_array, tile_size, stride_length)
    if alphabet is None or modify_alphabet:
        alphabet = create_alphabet(dequantizers, tiles, starter_alphabet)
    abstracted_array = create_abstracted_array(padded_array, tiles, alphabet)
    
    return dequantizers, tiles, abstracted_array, alphabet

# Logical blind bandit (inductive reasoning)
# bima_subs is Bijection Mask Substitutions (these are stored for applying Beamiter substitution sequences to get test output)
def modus_ponens(input_alphabet, input_matrix, abstracted_array, output_matrix, tile_size, P, batch_sub, batch_sub_tile_ind, bima_subs, delta_masks, alphabet_hash, fantasy_generator_map):
    # print(abstracted_array)
    # print(input_matrix)
    # print(output_matrix)
    for alpha_i, row in enumerate(abstracted_array[0]):
        for alpha_j, value in enumerate(row):
            if value == P:
                # print(f"Found {value} at {alpha_i}, {alpha_j}")
                # print(alphabet_hash[value][0]['tile'][0])
                # Now substitute
                i = alpha_i - (tile_size[1]-1)
                j = alpha_j - (tile_size[2]-1)
                # print(f"Subsitutute {value} at {i}, {j}")
                # for each value in output, if it is in bounds
                # print(alphabet_hash[value][0]['tile'][1])
                # print(delta_masks[value])
                has_mask = value in delta_masks
                mask = None
                if has_mask:
                    mask = delta_masks[value]
                    # print(mask)
                # Use either alphabet hash _or_ input_alphabet->fantasy generator
                
                # We have a letter value/P
                # Remember, that is just an undisproved by example i->o tile delta with a bitmask :)
                # Therefore, it would never be passed into Modus Ponens since we don't have a hash for proving 'imaginary' data ;)
                input_tile = alphabet_hash[value][0]['tile'][0]
                # print(input_tile)
                # print(input_tile)
                example_output_tile = alphabet_hash[value][0]['tile'][1]

                # Maybe make this a function...
                for y in range(tile_size[1]):
                    for x in range(tile_size[2]):
                        if (not has_mask or mask[y][x] == 1) and example_output_tile[y][x] != -1:
                            # output_matrix[0][i+y][j+x] = example_output_tile[y][x]
                            ind = (0, i+y, j+x)
                            if ind not in batch_sub:
                                if input_tile[y][x] != example_output_tile[y][x]: # only substitute if delta
                                    # print(f"{output_matrix[0][i+y][j+x]} became {example_output_tile[y][x]}")
                                    batch_sub[ind] = example_output_tile[y][x]
                                    # batch_sub_tile_ind[ind] = (0, alpha_i, alpha_j) # Map ind of pixel replacement to ind of tile in order to access local dequantizers
                                    print(value)
                                    batch_sub_tile_ind[ind] = value
                                    bima_subs[P] = (example_output_tile, mask)

# Fantasy crab (cyclic group finder)
def metacarcinus_magister():
    pass

def cast_spell(scene):
    # print(scene)
    # TODO: Try multiple tile sizes...
    tile_size = (2, 3, 3)
    # tile_size = (2, 2, 2)
    # tile_size = (2, 2, 2)

    quantization_hash = set([x for x in range(1,10)])

    task_id = scene

    stride_length = 1
    
    i = 0
    input_alphabet = {}
    io_alphabet = {}

    training_inputs = []

    while True:
        task = f"/home/heonae/arc/ide/save/arc/{task_id}/train/task_{i}/"
        if not os.path.exists(task) or "output" not in os.listdir(task):
            # print(i)
            break
        input_pattern = np.load(task + "input/grid.npy")
        output_pattern = np.load(task + "output/grid.npy")
        if input_pattern.shape != output_pattern.shape:
            return [] # TODO: We're going to have to work really hard to get this to work for different io sizes...
        io_pattern = np.stack([input_pattern, output_pattern])
        input_pattern = np.expand_dims(input_pattern, axis=0)
        training_inputs.append(input_pattern)
        # print(input_pattern.shape)
        input_dequantizers, input_tiles, input_abstracted_array, input_alphabet = abstract_array(quantization_hash, input_pattern, tile_size, stride_length, input_alphabet)
        # print("Input Alphabet Abstracted Array:\n", input_abstracted_array)

        io_dequantizers, io_tiles, io_abstracted_array, io_alphabet = abstract_array(quantization_hash, io_pattern, tile_size, stride_length, io_alphabet)
        # print("IO Alphabet Abstracted Array:\n", io_abstracted_array)
        i = i +1
    
    # TODO: Determine how alphabet_hash and delta_mask generation should be considered for test generalization

    # Alphabet hash maps input matrix tiles to list of corresponding io alphabet letters
    alphabet_hash = defaultdict(list)

    for io_letter in io_alphabet.items():
        tile_input_matrix = io_letter[1]['tile'][0]
        tile_input_matrix = np.expand_dims(tile_input_matrix, axis=0)
        # We use the input letters as the io alphabet letters list key...
        tile_key = f"{tile_input_matrix.shape}|{tile_input_matrix.tobytes()}"
        alphabet_hash[input_alphabet[tile_key]['letter']].append(io_letter[1])

    # Which bits delta is consistent across all io alphabet letters?
    delta_masks = {}
    delta_io = {}
    for input_letter, io_letters in alphabet_hash.items():
        # print(io_letters[0]['letter'])
        # print(len(letters))
        # Only generate masks for alphabet_hash len > 1
        if len(io_letters) > 1:
            # Get the input and output tiles
            input_tile = io_letters[0]['tile'][0]
            # Inputs are all the same, form delta on output!
            output_tiles = [letter['tile'][1] for letter in io_letters]
            # print(output_tiles)
            mask = np.all([np.equal(output_tiles[0], tile) for tile in output_tiles], axis=0).astype(int)
            # print(mask)
            delta_masks[input_letter] = mask.astype(int)
            io_different = 0

    symmetry_mapping, self_symmetry_mapping = create_symmetry_mapping(input_alphabet, [-1, 0], 'only')
    # print(list(symmetry_mapping.items())[0])
    # print(alphabet_hash[list(symmetry_mapping.items())[0][0]])
    generators, generator_map, find_generator = generate_finite_cyclical_groups(symmetry_mapping)
    # print(generators)
    # for generator in generators:
    #     for tile_key, letter_data in input_alphabet.items():
    #         if letter_data['letter'] == generator:
    #             print(letter_data)
    fantasy_generator_map = {}
    possbile_rots = set([1,2,3])
    # print(generators.keys())
    for gen, elements in generators.items():
        # print((gen, map))
        # print(map)
        # print(alphabet_hash[source_input][0]['tile'][0])
        existing_rots = set()
        for n in elements:
            z = generator_map[n][0]
            existing_rots.add(z)
        # print(existing_rots)
        if len(existing_rots) > 1:
            fantasy_rots_z_vals = possbile_rots.difference(existing_rots)
            for fantasy_z in fantasy_rots_z_vals:
                gen_input_tile = alphabet_hash[gen][0]['tile'][0].copy()
                # print(gen_input_tile)
                gen_input_tile = np.expand_dims(gen_input_tile, axis=0)
                # print(gen_input_tile[0])
                fantasy_tile = np.rot90(gen_input_tile, fantasy_z, axes=[1,2])
                # print(fantasy_tile[0])
                # Why is fantasy tile key letter not being found in test?
                # The reason is because test alphabets are not added to alphabet_hash
                # alphabet_hash cannot contain them because there is no io_alphabet for test
                # therefore, generate abstract array for only the input...
                fantasy_tile_key = f"{fantasy_tile.shape}|{fantasy_tile.tobytes()}"
                fantasy_generator_map[fantasy_tile_key] = (gen, fantasy_z) # inverse order since we're anchoring to generator letter...
    # We might only want to include fantasy mappings for cyclic groups which affect correctness and cause alphabet reduction
    # print(fantasy_generator_map.values())

    solved_training = test_training_hypothesis(quantization_hash, task_id, input_alphabet, tile_size, stride_length, alphabet_hash, delta_masks, fantasy_generator_map, False)
    
    # if solved_training:
    test_outputs = propose_test_outputs(training_inputs, quantization_hash, task_id, input_alphabet, tile_size, stride_length, alphabet_hash, delta_masks, fantasy_generator_map, False)
    # print(alphabet_hash['a'])
    return test_outputs
    # else:
    #     return []
        
    # else:
    #     print(f"Skipping {letters[0]['letter']} with only one letter")

# The alphabets can solve the training examples, however,
# can we reduce the alphabet size with 'equivalent colors' and symmetry rules?

def test_training_hypothesis(quantization_hash, task_id, input_alphabet, tile_size, stride_length, alphabet_hash, delta_masks, fantasy_generator_map, debug=False):
    i = 0
    solved_all_training = True
    training_outputs = []
    while True:
        if debug: print(f"Solved with alphabet of size {len(input_alphabet)}")
        task = f"/home/heonae/arc/ide/save/arc/{task_id}/train/task_{i}/"
        if not os.path.exists(task) or "output" not in os.listdir(task):
            if debug: print(i)
            break
        if debug: print("Train task")
        input_pattern = np.load(task + "input/grid.npy")
        output_pattern = np.load(task + "output/grid.npy")
        input_pattern = np.expand_dims(input_pattern, axis=0)
        output_pattern = np.expand_dims(output_pattern, axis=0)
        input_dequantizers, input_tiles, input_abstracted_array, input_alphabet = abstract_array(quantization_hash, input_pattern, tile_size, stride_length, input_alphabet)
        if debug: 
            print("Input Alphabet Abstracted Array:\n", input_abstracted_array)
            for row, row_letters in enumerate(input_abstracted_array[0]):
                for col, letter in enumerate(row_letters):
                    if letter in alphabet_hash:
                        if debug: 
                            print(len(alphabet_hash[letter]))
                            if len(alphabet_hash[letter]) == 9:
                                print(alphabet_hash[letter])
                            if letter in delta_masks:
                                print(delta_masks[letter])
        a = [letter for letter in alphabet_hash.keys()]
        hypothesis_output = input_pattern.copy()
        a.sort(key=lambda x: len(alphabet_hash[x]))
        last_n = a
        batch_sub = {}
        batch_sub_tile_ind = {}
        bima_subs = {}
        for v in last_n:
            modus_ponens(input_alphabet, input_pattern, input_abstracted_array, hypothesis_output, tile_size, v, batch_sub, batch_sub_tile_ind, bima_subs, delta_masks, alphabet_hash, fantasy_generator_map)
        for ind, val in batch_sub.items():
            hypothesis_output[ind[0]][ind[1]][ind[2]] = val
            # if debug: 
        # print(hypothesis_output)
        training_outputs.append(hypothesis_output)
        solved = output_pattern.size-np.sum((output_pattern!=hypothesis_output).astype(int))
        total = output_pattern.size
        if solved != total:
            solved_all_training = False
        print(f"{solved}/{total} correct")
        # if debug: print(f"Test {i}")
        
        i += 1
    return solved_all_training, training_outputs

def test_io_regen(quantization_hash, current_io, input_alphabet, tile_size, stride_length, alphabet_hash, delta_masks, fantasy_generator_map, debug=False):
    solved_all_training = True
    training_outputs = []
    for i, io_pattern in enumerate(current_io):
        input_pattern = io_pattern[0]
        output_pattern = io_pattern[1]
        input_pattern = np.expand_dims(input_pattern, axis=0)
        output_pattern = np.expand_dims(output_pattern, axis=0)
        print(io_pattern)
        input_dequantizers, input_tiles, input_abstracted_array, input_alphabet = abstract_array(quantization_hash, input_pattern, tile_size, stride_length, input_alphabet)
        if debug: 
            print("Input Alphabet Abstracted Array:\n", input_abstracted_array)
            for row, row_letters in enumerate(input_abstracted_array[0]):
                for col, letter in enumerate(row_letters):
                    if letter in alphabet_hash:
                        if debug: 
                            print(len(alphabet_hash[letter]))
                            if len(alphabet_hash[letter]) == 9:
                                print(alphabet_hash[letter])
                            if letter in delta_masks:
                                print(delta_masks[letter])
        a = [letter for letter in alphabet_hash.keys()]
        hypothesis_output = input_pattern.copy()
        a.sort(key=lambda x: len(alphabet_hash[x]))
        last_n = a
        batch_sub = {}
        batch_sub_tile_ind = {}
        bima_subs = {}
        for v in last_n:
            modus_ponens(input_alphabet, input_pattern, input_abstracted_array, hypothesis_output, tile_size, v, batch_sub, batch_sub_tile_ind, bima_subs, delta_masks, alphabet_hash, fantasy_generator_map)
        print("BIMA SUBS")
        print(bima_subs)
        for ind, val in batch_sub.items():
            hypothesis_output[ind[0]][ind[1]][ind[2]] = val
            # if debug: 
        # print(hypothesis_output)
        training_outputs.append(hypothesis_output)
        solved = output_pattern.size-np.sum((output_pattern!=hypothesis_output).astype(int))
        total = output_pattern.size
        if solved != total:
            solved_all_training = False
        print(f"{solved}/{total} correct")
        # if debug: print(f"Test {i}")

    return bima_subs, solved_all_training, training_outputs

"""
Given two task input matrices, identify the palette agnostic index alphabet underlying value mapping
ie
7 is like 4 because there are many equivalent quantized tiles or whatever
"""
def panalogize_dequanitization(quantization_hash, input_a, input_b, tile_size, stride_length):
    a_dequantizers, a_tiles, a_abstracted_array, a_alphabet = abstract_array(quantization_hash, input_a, tile_size, stride_length, None)
    b_dequantizers, b_tiles, b_abstracted_array, b_alphabet = abstract_array(quantization_hash, input_b, tile_size, stride_length, None)
    panalogy = {}
    for tile_key, letter in a_alphabet.items():
        if tile_key in b_alphabet:
            # print(f"a_alphabet_{letter['letter']} is analogous to b_alphabet_{b_alphabet[tile_key]['letter']}")
            if len(letter['deq']) > 0:
                # print(f"{letter['deq']} is analogous to {b_alphabet[tile_key]['deq']}")
                b_deq_map = b_alphabet[tile_key]['deq']
                for quant_index, real_val in letter['deq'].items():
                    # Both of these are valid panalogies
                    # panalogy[quant_index] = (real_val, b_deq_map[quant_index])
                    panalogy[real_val] = b_deq_map[quant_index]
            # TODO: Per tile panalogy or task global?
    return panalogy

def propose_test_outputs(training_inputs, quantization_hash, task_id, input_alphabet, tile_size, stride_length, alphabet_hash, delta_masks, fantasy_generator_map, debug=False):
    test_outputs = []
    i = 0
    while True:
        if debug: print(f"Solved with alphabet of size {len(input_alphabet)}")
        task = f"/home/heonae/arc/ide/save/arc/{task_id}/test/task_{i}/"
        if not os.path.exists(task) or "output" not in os.listdir(task):
            if debug: print(i)
            break
        if debug: print("Test task")
        input_pattern = np.load(task + "input/grid.npy")
        output_pattern = np.load(task + "output/grid.npy")
        input_pattern = np.expand_dims(input_pattern, axis=0)
        output_pattern = np.expand_dims(output_pattern, axis=0)
        input_dequantizers, input_tiles, input_abstracted_array, input_alphabet = abstract_array(quantization_hash, input_pattern, tile_size, stride_length, input_alphabet)
        if debug: 
            print("Input Alphabet Abstracted Array:\n", input_abstracted_array)
            for row, row_letters in enumerate(input_abstracted_array[0]):
                for col, letter in enumerate(row_letters):
                    if letter in alphabet_hash:
                        if debug: 
                            print(len(alphabet_hash[letter]))
                            if len(alphabet_hash[letter]) == 9:
                                print(alphabet_hash[letter])
                            if letter in delta_masks:
                                print(delta_masks[letter])
        a = [letter for letter in alphabet_hash.keys()]
        hypothesis_output = input_pattern.copy()
        a.sort(key=lambda x: len(alphabet_hash[x]))
        # So again, what is v?
        # It is an undisproved by example i->o tile delta mask alphabet hash key
        last_n = a
        batch_logic_sub = {}
        batch_sub_tile_ind = {}
        bima_subs = {}
        for v in last_n:
            modus_ponens(input_alphabet, input_pattern, input_abstracted_array, hypothesis_output, tile_size, v, batch_logic_sub, batch_sub_tile_ind, bima_subs, delta_masks, alphabet_hash, fantasy_generator_map)
        for ind, val in batch_logic_sub.items():
            # We need to get the local dequantizer for this tile sub specifically :|
            # print("TODO: Get io_dequantizer from input tile")
            print(alphabet_hash[batch_sub_tile_ind[ind]])
            deq = alphabet_hash[batch_sub_tile_ind[ind]][0]['deq']
            sub_val = val
            if val in deq:
                print("value dequantized")
                sub_val = deq[val]
                # Get deq from tile
            hypothesis_output[ind[0]][ind[1]][ind[2]] = sub_val
            # print(input_dequantizers[batch_sub_tile_ind[ind]]) # TODO: Use io_dequantizers
            if debug: 
                print(hypothesis_output)
                print(f"{output_pattern.size-np.sum((output_pattern!=hypothesis_output).astype(int))}/{output_pattern.size} correct")
        
        # Follow up Modus Ponens with a fantasy attack, Magi ;)

        batch_fantasy_sub = {}
        for tile_coords, tile in input_tiles.items():
            # print(tile)
            input_tile_key = f"{tile.shape}|{tile.tobytes()}"
            if input_tile_key in fantasy_generator_map:
                # print("To generalize, you'll need more than inductive data, Modus. Here, I scavenged a fantasy cyclic group element for you! Sorry it's just 64 bytes...")
                # Now substitute, don't worry about batch for now...
                # print(tile)
                generator = fantasy_generator_map[input_tile_key][0]
                rot_z = fantasy_generator_map[input_tile_key][1]
                gen_tile_data = alphabet_hash[generator]
                gen_tile = gen_tile_data[0]['tile'][1]
                # print(gen_tile[0]['tile'][1])
                gen_tile = np.expand_dims(gen_tile, axis=0)
                output = np.rot90(gen_tile, rot_z, axes=[1,2])[0]
                has_mask = generator in delta_masks
                if has_mask:
                    gen_mask = delta_masks[generator]
                    # print(gen_mask)
                    output_mask = np.rot90(np.expand_dims(gen_mask, axis=0), rot_z, axes=[1,2])[0]
                    # print("FOUND OUTPUT MASK")
                    # print(output_mask)

                # DO THE SUBSTITUTION!
                for y in range(tile_size[1]):
                    for x in range(tile_size[2]):
                        if (not has_mask or output_mask[y][x] == 1) and output[y][x] != -1:
                            pass
                            tile_row_index = tile_coords[1] - (tile_size[1]-1)
                            tile_col_index = tile_coords[2] - (tile_size[1]-1)
                            ind = (0, tile_row_index+y, tile_col_index+x)
                            batch_fantasy_sub[ind]= output[y][x]
        
        for ind, val in batch_fantasy_sub.items():
            hypothesis_output[ind[0]][ind[1]][ind[2]] = val

        # Panalogize fantasy even wow!?
            
        # Which of the training inputs should we use to panalogize?
        # This will affect correctness as in Panalogy Sniper
        panalogy = panalogize_dequanitization(quantization_hash, training_inputs[1], input_pattern, tile_size, stride_length)
        for y in range(hypothesis_output.shape[1]):
            for x in range(hypothesis_output.shape[2]):
                val = hypothesis_output[0][y][x]
                if val in panalogy:
                    hypothesis_output[0][y][x] = panalogy[val]

        if debug: print(f"Test {i}")
        test_outputs.append(hypothesis_output)
        i += 1
    return test_outputs


# Print alphabet with shape information
# for tile_key, tile_info in alphabet.items():
#     print(f"Letter: {tile_info['letter']}, Shape: {tile_info['shape']}, Tile:\n{tile_info['tile']}")

# import networkx as nx
# import matplotlib.pyplot as plt

def create_symmetry_mapping(alphabet, filter_values=None, filter_mode='any'):
    symmetry_mapping = defaultdict(list)
    self_symmetry_mapping = {}
    
    for tile_key, tile_info in alphabet.items():
        tile = tile_info['tile']
        
        if filter_values is not None:
            if filter_mode == 'any' and np.any(np.isin(tile, filter_values)):
                continue
            elif filter_mode == 'only':
                filter_out = True
                for value in tile.flatten():
                    if value not in filter_values:
                        filter_out = False
                if filter_out:
                    continue
        
        # Flipping horizontally
        # flipped_tile_h = np.fliplr(tile[0])
        # flipped_tile_h_key = f"{flipped_tile_h.shape}|{flipped_tile_h.tobytes()}"
        # if flipped_tile_h_key in alphabet:
        #     if tile_key == flipped_tile_h_key:
        #         self_symmetry_mapping[tile_info['letter']] = 'horizontal_flip'
        #     else:
        #         symmetry_mapping[tile_info['letter']].append(('horizontal_flip', alphabet[flipped_tile_h_key]['letter']))
        
        # # Flipping vertically
        # flipped_tile_v = np.flipud(tile[0])
        # flipped_tile_v_key = f"{flipped_tile_v.shape}|{flipped_tile_v.tobytes()}"
        # if flipped_tile_v_key in alphabet:
        #     if tile_key == flipped_tile_v_key:
        #         self_symmetry_mapping[tile_info['letter']] = 'vertical_flip'
        #     else:
        #         symmetry_mapping[tile_info['letter']].append(('vertical_flip', alphabet[flipped_tile_v_key]['letter']))
        
        # Rotating 90 degrees clockwise
        rotated_tile_90 = np.rot90(tile, axes=[1,2])
        rotated_tile_90_key = f"{rotated_tile_90.shape}|{rotated_tile_90.tobytes()}"
        if rotated_tile_90_key in alphabet:
            if tile_key == rotated_tile_90_key:
                self_symmetry_mapping[tile_info['letter']] = 'rotation_90'
            else:
                symmetry_mapping[tile_info['letter']].append((1, alphabet[rotated_tile_90_key]['letter']))
        
        # Rotating 180 degrees
        rotated_tile_180 = np.rot90(tile, 2, axes=[1,2])
        rotated_tile_180_key = f"{rotated_tile_180.shape}|{rotated_tile_180.tobytes()}"
        if rotated_tile_180_key in alphabet:
            if tile_key == rotated_tile_180_key:
                self_symmetry_mapping[tile_info['letter']] = 'rotation_180'
            else:
                symmetry_mapping[tile_info['letter']].append((2, alphabet[rotated_tile_180_key]['letter']))
        
        # Rotating 270 degrees (equivalent to -90 degrees)
        rotated_tile_270 = np.rot90(tile, 3, axes=[1,2])
        rotated_tile_270_key = f"{rotated_tile_270.shape}|{rotated_tile_270.tobytes()}"
        if rotated_tile_270_key in alphabet:
            if tile_key == rotated_tile_270_key:
                self_symmetry_mapping[tile_info['letter']] = 'rotation_270'
            else:
                symmetry_mapping[tile_info['letter']].append((3, alphabet[rotated_tile_270_key]['letter']))
    
    return symmetry_mapping, self_symmetry_mapping

def generate_finite_cyclical_groups(symmetry_mapping):
    generators = {}
    generator_map = {}
    find_generator = {}
    for letter, transforms in symmetry_mapping.items():
        if letter in generator_map:
            pass
        else:
            generators[letter] = []
            for transform in transforms:
                generator_map[transform[1]] = (transform[0], letter)
                generators[letter].append(transform[1])
                find_generator[letter] = transform[1]
    return generators, generator_map, find_generator

# test_training_solve(input_alphabet)
# TODO: test each letter in input alphabet on the training examples, identify the ones with the highest positive 'correctness' delta
# Eliminate letters which have no or negative change on correctness 
# Use cyclical groups and color palettes to further reduce alphabet

# symmetry_mapping, self_symmetry_mapping = create_symmetry_mapping(input_alphabet, [-1, 0], 'only')
# print(list(symmetry_mapping.items())[0])
# generators, generator_map, find_generator = generate_finite_cyclical_groups(symmetry_mapping)
# for generator in generators:
#     for tile_key, letter_data in input_alphabet.items():
#         if letter_data['letter'] == generator:
#             print(letter_data)



# def find_recurring_subgraphs(symmetry_mapping, frequency_threshold=2):
#     G = nx.Graph()
#     for (letter1, letter2), operation in symmetry_mapping.items():
#         G.add_edge(letter1, letter2, operation=operation)

#     subgraphs = []
#     for c in nx.connected_components(G):
#         subgraph = G.subgraph(c)
#         subgraph_hash = hash_subgraph(subgraph)
#         subgraphs.append((subgraph, subgraph_hash))
#         print(subgraph)

#     subgraph_counts = {}
#     for _, subgraph_hash in subgraphs:
#         if subgraph_hash not in subgraph_counts:
#             subgraph_counts[subgraph_hash] = 0
#         subgraph_counts[subgraph_hash] += 1

#     recurring_subgraphs = []
#     for subgraph, subgraph_hash in subgraphs:
#         if subgraph_counts[subgraph_hash] >= frequency_threshold:
#             recurring_subgraphs.append(subgraph)

#     return recurring_subgraphs

# def hash_subgraph(subgraph):
#     edge_features = []
#     for u, v in subgraph.edges():
#         edge_features.append(subgraph[u][v]['operation'])
#     return tuple(sorted(edge_features))

# def create_composition_mapping(alphabet):
#     composition_mapping = {}
#     for tile_key1, tile_info1 in alphabet.items():
#         for tile_key2, tile_info2 in alphabet.items():
#             if tile_info1['shape'] == tile_info2['shape']:
#                 tile1 = tile_info1['tile']
#                 tile2 = tile_info2['tile']
#                 # Checking for composition by replacing 1's from tile1 and tile2
#                 composed_tile = np.where(tile1 == 1, tile2, tile1)
#                 composed_tile_key = f"{composed_tile.shape}|{composed_tile.tobytes()}"
#                 if composed_tile_key in alphabet:
#                     composition_mapping[(tile_info1['letter'], tile_info2['letter'])] = alphabet[composed_tile_key]['letter']
    
#     return composition_mapping

# symmetry_mapping = create_symmetry_mapping(alphabet, [-1, 0], 'only')
# recurring_subgraphs = find_recurring_subgraphs(symmetry_mapping, frequency_threshold=3)
# for subgraph in recurring_subgraphs:
#     print(subgraph.nodes(), subgraph.edges(data=True))
# composition_mapping = create_composition_mapping(alphabet)

# print("Symmetry Mapping:")
# for (letter1, letter2), symmetry_type in symmetry_mapping.items():
#     print(f"{letter1} and {letter2} are symmetric by {symmetry_type}")

# print("\nComposition Mapping:")
# for (letter1, letter2), composed_letter in composition_mapping.items():
#     pass
#     # print(f"{letter1} and {letter2} compose to {composed_letter}")

# G = nx.Graph()
# for (letter1, letter2), symmetry_type in symmetry_mapping.items():
#     G.add_edge(letter1, letter2, label=symmetry_type)

# pos = nx.spring_layout(G, k=0.8)  
# nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
# nx.draw_networkx_edges(G, pos, width=1, edge_color='gray')
# nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# labels = nx.get_edge_attributes(G, 'label')

# for edge, label in labels.items():
#     if edge[0] == edge[1]:  # Self-edge
#         x, y = pos[edge[0]]
#         plt.text(x, y + 0.1, label, ha='center', va='bottom', fontsize=8)
#     else:
#         nx.draw_networkx_edge_labels(G, pos, edge_labels={edge: label}, font_size=8, label_pos=0.5)

# plt.axis('off')  
# plt.figure(figsize=(10, 8))  
# plt.show()

if __name__ == "__main__":
    arc_tasks_path =f"/home/heonae/arc/ide/save/arc/"

    # Use object priors to isolate alphabets
    # scene = "3befdf3e" # Fetid Flower Spawner

    # TODO: Combine quantized panalogy with beamiter to fire the laser...
    # scene = "25d487eb" # Spaceship Beamiter
    scene = "67a423a3" # Panalogy Sniper

    # A simple beamiter panalogy
    # scene = "7ddcd7ec" # Fantasy Test Canceled

    # TODO: TOMORROW
    # Mathematics or Gender Studies?
    # 8d510a79

    # Easy
    # scene = "0962bcdd"

    task_id = scene
    print(scene)
    test_outputs = cast_spell(scene)
    print(test_outputs)
    arc_scene = os.path.join(arc_tasks_path, scene)
    if test_outputs and len(test_outputs) > 0:
        for i, proposal in enumerate(test_outputs):
            task = os.path.join(arc_scene, f"test/task_{i}/")
            output_pattern = np.load(task + "output/grid.npy")
            output_pattern = np.expand_dims(output_pattern, axis=0)
            solved_values = output_pattern.size-np.sum((output_pattern!=proposal).astype(int))
            total_values = output_pattern.size
            print(f"{solved_values}/{total_values} correct")
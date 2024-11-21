# Alphabet Hash Bijection Mask Beam Iterator

"""
Alhabima Beamiter is an iterative cast of Alphabet Hash Bijection Mask (involving alphabet extraction at each iteration)
Such that a problem can be solved iteratively (good examples are Harvard Garden and Bedrock Blaster)
"""

from collections import defaultdict
import numpy as np
import string
import os
import io_alhabima as alhabima

def cast_spell(scene):
    
    # TODO: Try multiple tile sizes...
    # tile_size = (2, 3, 3)
    tile_size = (2, 3, 3)
    # tile_size = (2, 2, 2)

    task_id = scene

    stride_length = 1
    
    input_alphabet = {}
    io_alphabet = {}

    task_outputs = []
    current_io = []

    # Initial loading of tasks
    i = 2
    while True:
        task = f"/home/wesxdz/arc/ide/save/arc/{task_id}/train/task_{i}/"
        if not os.path.exists(task) or "output" not in os.listdir(task):
            # print(i)
            break
        input_pattern = np.load(task + "input/grid.npy")
        output_pattern = np.load(task + "output/grid.npy") # Remember, you are not allowed to load test outputs :)
        task_outputs.append(output_pattern)
        if input_pattern.shape != output_pattern.shape:
            return [] # TODO: We're going to have to work really hard to get this to work for different io sizes...
        io_pattern = np.stack([input_pattern, output_pattern])
        current_io.append(io_pattern)
        break # Just test on one
        i = i +1

    max_beam = 10
    bima_sub_sequences = []
    for beam in range(max_beam):
        bima_sub_sequences.append([])

        # From the beam iterator, we should gather a 'sequence of substitutions' used to solve training examples...
        # This is similar to batch_sub parameter in modus ponens
        # We want to store the value/mask that was used

        for io_pattern in current_io:
            input_pattern = np.expand_dims(io_pattern[0], axis=0)
            input_tiles, input_abstracted_array, input_alphabet = alhabima.abstract_array(input_pattern, tile_size, stride_length, input_alphabet)
            # print("Input Alphabet Abstracted Array:\n", input_abstracted_array)

            io_tiles, io_abstracted_array, io_alphabet = alhabima.abstract_array(io_pattern, tile_size, stride_length, io_alphabet)
            # print("IO Alphabet Abstracted Array:\n", io_abstracted_array)
        
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

        symmetry_mapping, self_symmetry_mapping = alhabima.create_symmetry_mapping(input_alphabet, [-1, 0], 'only') # maybe remove 0 filter...
        # print(list(symmetry_mapping.items())[0])
        # print(alphabet_hash[list(symmetry_mapping.items())[0][0]])
        generators, generator_map, find_generator = alhabima.generate_finite_cyclical_groups(symmetry_mapping)
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
                    fantasy_generator_map[fantasy_tile_key] = (gen, fantasy_z)

        bima_subs, solved_training, training_hypothesis_outputs = alhabima.test_io_regen(current_io, input_alphabet, tile_size, stride_length, alphabet_hash, delta_masks, fantasy_generator_map, False)
        # print(solved_training)
        # For each training task, the hypothesis output will be recast in Alhabima as the input
        # What is the halting condition? Training solved or
        regenerated_io = []
        # print(len(training_hypothesis_outputs))
        for o_i in range(len(training_hypothesis_outputs)):
            # print(training_hypothesis_outputs[o_i][0])
            # print(task_outputs[o_i])
            regenerated_io.append(np.stack([training_hypothesis_outputs[o_i][0], task_outputs[o_i]]))
        # print(regenerated_io)
        if len(bima_subs) == 0:
            break # No more substitutions
        else:
            bima_sub_sequences[beam].extend(bima_subs.items())

        current_io = regenerated_io

    print(bima_sub_sequences)
    # Initial loading of tasks
    test_outputs = []
    i = 0
    while True:
        task = f"/home/wesxdz/arc/ide/save/arc/{task_id}/test/task_{i}/"
        if not os.path.exists(task) or "output" not in os.listdir(task):
            # print(i)
            break
        input_pattern = np.load(task + "input/grid.npy")
        # output_pattern = np.load(task + "output/grid.npy") # Remember, you are not allowed to load test outputs in spells :)
        input_pattern = np.expand_dims(input_pattern, axis=0)
        test_outputs.append(input_pattern)
        i = i +1

    for bima_sub_wave in bima_sub_sequences:
        for output in test_outputs:
            for beam in range(5):
                # When applying the beam more than once, we need to be really careful not to make mistakes
                # by substituting a region falsely
                print("BIMA SUB WAVE")
                print(bima_sub_wave)
                batch_sub = {}
                for s_i, sub in enumerate(bima_sub_wave):
                    print(s_i)
                    print()
                    # if sub is found in the test input, replace it on the wave hahaha!
                    # output is input ;)
                    input_tiles, input_abstracted_array, input_alphabet = alhabima.abstract_array(output, tile_size, stride_length, input_alphabet)
                    print(input_abstracted_array)
                    break
                    
                    sub_output = sub[1][0]
                    sub_mask = sub[1][1]
                    print(sub_mask)
                    sub_letter = sub[0]
                    print()
                    sub_input = alphabet_hash[sub_letter][0]['tile'][0]
                    print(sub_input)
                    print("Sub output")
                    print(sub_output)
                    sub_input = np.expand_dims(sub_input, axis=0)
                    # print(sub_mask)
                    sub_key = f"{sub_input.shape}|{sub_input.tobytes()}"
                    if sub_key in input_alphabet:
                        print("Found key")
                        print(input_alphabet[sub_key])
                    else:
                        print("No key")
            
                    for alpha_i, row in enumerate(input_abstracted_array[0]):
                        for alpha_j, value in enumerate(row):
                            if value == sub_letter:
                            # pass
                            #     # print(f"Found {value} at {alpha_i}, {alpha_j}")
                            #     # print(alphabet_hash[value][0]['tile'][0])
                            #     # Now substitute
                                row_index = alpha_i - (tile_size[1]-1)
                                col_index = alpha_j - (tile_size[2]-1)
                                for y in range(tile_size[1]):
                                    for x in range(tile_size[2]):
                                        valid_mask = True
                                        if sub_mask is not None:
                                            valid_mask = sub_mask[y][x]
                                        if sub_output[y][x] != -1 and valid_mask:
                                            ind = (0, row_index+y, col_index+x)
                                            if ind not in batch_sub:
                                                print(col_index+x)
                                                batch_sub[ind] = sub_output[y][x]
                for sub_ind, sub_val in batch_sub.items():
                    output[sub_ind[0]][sub_ind[1]][sub_ind[2]] = sub_val
    return test_outputs
    
    # test_outputs = alhabima.propose_test_outputs(task_id, input_alphabet, tile_size, stride_length, alphabet_hash, delta_masks, fantasy_generator_map, True)
    # return test_outputs


if __name__ == "__main__":
    arc_tasks_path =f"/home/wesxdz/arc/ide/save/arc/"

    scene = "28e73c20" # Harvard Garden, merging bima subs indiscriminately causes problems here
    # scene = "d4f3cd78"
    # scene = "7ddcd7ec"
    # scene = "5c0a986e"

   #scene = "2bee17df" # Santa Caved In

    # scene = "aba27056" # Palette Quantizer Gun
    # We need to create a panalogy which equates yellow/slate, and maps the other color to a palette index
    # The question is how to determine this panalogy?
    # For now we can simply iterate all combinations as potential quantizations
    # and determine anchoring by the training solve...

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
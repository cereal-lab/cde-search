''' DECA algorithm for dimension extraction. Thanks to Dr. Paul Wiegand 
    https://github.com/cereal-lab/EvoPIE/blob/dashboard-integration/evopie/datadashboard/analysislayer/deca.py

    Bucci, Pollack, & de Jong (2004).  "Automated Extraction
    of Question Structure".  In Proceedings of the 2004 Genetic
    and Evolutionary Computation Conference.  

    The algorithm is modified in next manner:
       1. Spanned point is the points that can combined 2 or MORE axes (in contract to only 2 in original article)  
       2. Origin, spanned and duplicates are collected separatelly 
             TODO: they should be extanded with additional info: spanned - what [(axis_id, point_id)...] where combined
                                                                 duplicates - what test is already represents it and what (axis_id, point_id)
    As with original algo, there could be situation when new test can extend several axes - in this case first axis is selected
        - this is what makes the algo depending on order of tests, probably sorting at start should fixate the order more strictly
'''

def extract_dims(tests):
    origin = []
    spanned = []
    dimensions = []    
    duplicates = []

    for test_id, test in sorted(enumerate(tests), key=lambda x: sum(x[1])):        
        if all(t == 0 for t in test): #origin 
            origin.append(test_id)
            continue
        test_dims = []
        is_dup = False
        for dim in dimensions:
            if all(t == d for t, d in zip(test, dim[-1][1])):
                duplicates.append(test_id)
                is_dup = True
                break 
            else:
                i = len(dim) - 1
                while i >= 0:
                    if all(t >= d for t, d in zip(test, dim[i][1])) and any(t > d for t, d in zip(test, dim[i][1])):
                        test_dims.append((dim, i))   
                        break 
                    i -= 1
        if is_dup:
            continue
        all_ands = [max(el) for el in zip(*[dim[dim_pos][1] for (dim, dim_pos) in test_dims])]
        if all_ands == test: #spanned 
            spanned.append(test_id)
            continue 
        elif len(all_ands) == 0: #extend any of dims 
            dimensions.append([(test_id, test)])
        else:
            at_ends_dims = [dim for dim, pos in test_dims if pos == len(dim) - 1]
            if len(at_ends_dims) > 0:
                at_ends_dims[0].append((test_id, test)) 
            else:
                dimensions.append([(test_id, test)])

    dims = [[i for i, _ in dim] for dim in dimensions]
    return dims, origin, spanned, duplicates
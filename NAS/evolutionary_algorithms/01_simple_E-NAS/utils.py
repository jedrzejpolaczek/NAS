from bitstring import BitArray


def convert_list_to_int(bit_list):
    bit_string = ''.join(str(bit) for bit in bit_list)
 
    number = BitArray(bin=bit_string).uint

    return number

def individual_guard(individual):
    if type(individual) is list:
        if type(individual[0]) is list:
            if type(individual[1]) is list:
                return True
    
    return False

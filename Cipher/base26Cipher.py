import json

BASE=26

def save_to_json(obj, filename):
    obj_string = json.dumps(obj)
    with open(filename, 'w') as f:
        json.dump(obj_string, f)

def read_json(filename):
    with open(filename, 'r') as f:
        obj_string = json.load(f)
    obj = json.loads(obj_string)
    return obj 

def load_encoder():
    encoder_path = f"./Cipher/base{BASE}encoder.json" 
    encoder = read_json(encoder_path)
    return encoder

def load_decoder():
    decoder_path = f"./Cipher/base{BASE}decoder.json"
    decoder = read_json(decoder_path)
    return decoder
def load_encoder_decoder():
    """
        Return Encoder and Decoder from .json
    Returns:
        encoder, decoder : Object
    """
    encoder = load_encoder()
    decoder = load_decoder()
    return encoder, decoder



def encode_word(word):
    encoder = load_encoder()
    word_len = len(word)
    result = 0
    for indx, char in enumerate(word):
        p = word_len - indx - 1
        result += int(encoder[char])*26**p
    return result

def decode_word(number):
    decoder = load_decoder()
    result = ""
    while number // BASE or number % (BASE):
        letter_code = number % (BASE)
        number = number // (BASE)
        if letter_code == 0:
            number -= 1
            letter_code += 1*26
        result = decoder[str(letter_code)] + result 
        
    return result


if __name__ == '__main__':

    word = "hello"

    number = encode_word(word)
    print(decode_word(number) == word)
    


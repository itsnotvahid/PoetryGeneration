import torch
from torch.nn import functional as F

from configs import configs
from data import vocab, index2char


def generate(model, prompt, temp, many):
    model.eval()
    input_eval = [vocab[c] for c in prompt]
    with torch.no_grad():
        for i in range(many):
            model.eval()
            predictions = model(torch.LongTensor(input_eval).unsqueeze(0).to(configs['device']))
            predictions = (predictions.squeeze() / temp)
            arg = torch.multinomial(F.softmax(predictions, dim=-1), 1)[-1]
            input_eval.append(arg)
    text = ''.join(index2char[c] for c in input_eval)
    return text

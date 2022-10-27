import re
from collections import Counter

def AAC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []

    for sequence in sequences:
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings


def DPC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []

    DiPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    AADict = dict()
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for sequence in sequences:
        code = [0] * 400
        for i in range(len(sequence) - 1):
            code[AADict[sequence[i]] * 20 + AADict[sequence[i+1]]] = code[AADict[sequence[i]] * 20 + AADict[sequence[i+1]]] + 1
        if sum(code) != 0:
            code = [code[i] / sum(code) for i in code]
        encodings.append(code)
    return encodings


def TPC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []

    TriPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
    AADict = dict()
    for i in range(len(AA)):
        AADict[AA[i]] = i
    
    for sequence in sequences:
        code = [0] * 8000
        for i in range(len(sequence) - 2):
            code[AADict[sequence[i]] * 400 + AADict[sequence[i+1]] * 20 + AADict[sequence[i+2]]] = code[AADict[sequence[i]] * 400 + AADict[sequence[i+1]] * 20 + AADict[sequence[i+2]]] + 1
        if sum(code) !=0:
            code = [code[i] / sum(code) for i in code]
        encodings.append(code)
    return encodings
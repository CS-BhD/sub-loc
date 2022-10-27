from gensim.models import doc2vec
import numpy as np

def seq_to_k_sentence(seq, k=3, overlop=False):
    """ Divide a sequence into k sentence

    Parameters:
        seq(string): sequence
        k(int): k grams
        overlop(bool): whether to overlop, default = False
    
    Return:
        containing a list 1 sentence (if overlop = True),
        or a list of k sentences (overlop = False)
    """
    l = len(seq)
    if overlop:
        return [[seq[i:i+k] for i in range(l-k+1)]]
    else:
        return [[seq[i:i+k] for i in range(j, l-k+1, k)] for j in range(k)]


def seqs_to_k_sentence(seqs, k=3, overlop=False):
    """ Divide a list of sequences into k sentences

    Parameters:
        seqs(list): a list of sequence(string)
        k(int): k grams
        overlop(bool): whether to overlop, default = False
    
    Returns:
        List of lists of k sentences
    """
    documents = []
    for seq in seqs:
        documents += seq_to_k_sentence(seq, k=k, overlop=overlop)
    return documents


# def get_vectors(sequences, model_file, k):
#     """Infer document vector from model
#     Parameters:
#         sequences (list): list of protein sequence
#         model_file (WindowPath): model path
#         k (int): divide one sequence into k grams
#     Return: 
#         vectors (ndarray): ndarray of (sequence -> sequence embedding)
#     """
#     vectors = []
#     model = doc2vec.Doc2Vec.load(str(model_file))
#     for seq in sequences:
#         sentences = seq_to_k_sentence(seq, k=k)
#         vector = np.array([model.infer_vector(sentence) for sentence in sentences])
#         vectors.append(vector.mean(0))
#     return np.array(vectors)

    # 子句的向量加和
def getVecs_mean(model, sequences, k, mean=True):
    vectors = []
    for sequence in sequences:
        sentences = seq_to_k_sentence(sequence, int(k))
        vector = np.array([model.infer_vector(sentence) for sentence in sentences])
        if mean is True:
            vectors.append(vector.mean(0))
        else:
            vectors.append(vector.sum(0))
    return vectors

# 将每个子句的向量链接在一起
def getVecs(model, sequences, k, mean=True):
    vectors = []
    for sequence in sequences:
        sentences = seq_to_k_sentence(sequence, int(k))
        vector = []
        for sentence in sentences:
            vector.extend(model.infer_vector(sentence))
        vectors.append(vector)
    return np.array(vectors)


def get_vectors(dm, dbow, sequences, k, mean=True):
    dm_vecs = getVecs(dm, sequences, k, mean)
    dbow_vecs = getVecs(dbow, sequences, k, mean)
    vecs = np.concatenate((dm_vecs, dbow_vecs), axis=1)
    return vecs

def sequence_to_vec(sequence, wv, k):
    """Infer sequence vector from keyedvector
    Parameters:
        sequence (str): protein sequence
        wv(KeyedVector): word vector
        k (int): divide one sequence into k grams
    Return:
        vectors (ndarray): array of (word -> vector)
    """
    vectors = []
    sentences = seq_to_k_sentence(sequence, k=k)
    for sentence in sentences:
        sen_vector = []
        print(len(sentence))
        for word in sentence:
            vector = wv.get_vector(word)
            sen_vector.append(vector)
        vectors.append(np.array(sen_vector))
    return np.array(vectors)
            

class Corpus(object):

    def __init__(self, sequences, hypers):
        self.sequences = sequences
        self.hypers = hypers


    def __iter__(self):
        for doc in self.get_document():
            yield doc
        
    
    def seq_to_sentences(self):
        for seq in self.sequences:
            grams = seq_to_k_sentence(seq, k = self.hypers['k'], overlop=self.hypers['overlop'])
            if self.hypers['overlop']:
                yield grams
            else:
                for gram in grams:
                    yield gram
            

    def get_document(self):
        if self.hypers['merge']:
            return (doc2vec.TaggedDocument(doc, [i // self.hypers['k']]) for i, doc in enumerate(self.seq_to_sentences()))
        else:
            return (doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(self.seq_to_sentences()))
            

class Sen_Corpus(object):

    def __init__(self, sequences, hypers):
        super().__init__()
        self.sequences = sequences
        self.hypers = hypers

    
    def __iter__(self):
        for sen in self.seq_to_sentences():
            yield sen

    def seq_to_sentences(self):
        for seq in self.sequences:
            grams = seq_to_k_sentence(seq, k = self.hypers['k'], overlop=self.hypers['overlop'])
            if self.hypers['overlop']:
                yield grams[0]
            else:
                for gram in grams:
                    yield gram
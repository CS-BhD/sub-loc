from Bio import SeqIO
import re
def read_protein_file(file, format='fasta', label=None):
    """Read protein file
    Parameter:
        file (str): file path
        format (str): file format. Default 'fasta'
        label (str): protein location
    Return:
        list of [protein_name, sequence, location]
    """
    records = []
    for record in SeqIO.parse(file, format):
        records.append([record.id.split('|')[1], str(record.seq), label])
    return records


def read_and_replace_protein_file(file, format='fasta', label=None):
    """Read protein file and utilize '-' to replace "[BJOUXZ]"
    Parameter:
        file (str): file path
        format (str): file format. Default 'fasta'
        label (str): protein location
    Return:
        list of [protein_name, sequence, location]
    """
    records = []
    for record in SeqIO.parse(file, format):
        records.append([record.id.split('|')[1], re.sub('[BJOUXZ]', '-',str(record.seq)), label])
    return records

def read_and_remove_protein_file(file, format='fasta', label=None):
    """Read protein file and remove the sequence which contains "[BJOUXZ]"
    Parameter:
        file (str): file path
        format (str): file format. Default 'fasta'
        label (str): protein location
    Return:
        list of [protein_name, sequence, location]
    """
    records = []
    for record in SeqIO.parse(file, format):
        if re.search('[BJOUXZ]', str(record.seq)) is None:
            records.append([record.id.split('|')[1], str(record.seq), label])
    return records

def remove_fasta_description(read_path, write_path, format='fasta'):
    """Remove the description of protein sequence in fasta format file
    Parameter:
        read_path(str): fasta file save protein sequence
        write_path (str): file path to write
        format (str): file format. Default 'fasta'
    Return:
        the number of written protein sequences
    """
    records = []
    for record in SeqIO.parse(read_path, format):
        record.description = ''
        records.append(record)
    return SeqIO.write(records, write_path, format)
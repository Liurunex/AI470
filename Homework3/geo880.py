import collections
import os
import random
random.seed(0)

IN_FILE = '/c/cs570/data/geoquery/geoqueries880'
OUT_DIR = './data/geo880/'
#IN_FILE = 'sample_input.txt'
#OUT_DIR = '.'
if not os.path.exists(OUT_DIR):
  os.makedirs(OUT_DIR)

# TO ADD space in the logic form
def split_logical_form(lf):
  replacements = [
      ('(', ' ( '),
      (')', ' ) '),
      (',', ' , '),
      ("'", " ' "),
      ("\\+", " \\+ "),
  ]
  for a, b in replacements:
    lf = lf.replace(a, b)
  return ' '.join(lf.split())

def reduce_copying(lf):
  # List all predicates (whitelist)
  PREDS = [
      'cityid', 'countryid', 'placeid', 'riverid', 'stateid',
      'capital', 'city', 'lake', 'major', 'mountain', 'place', 'river',
      'state', 'area', 'const', 'density', 'elevation', 'high_point',
      'higher', 'loc', 'longer', 'low_point', 'lower', 'len', 'next_to',
      'population', 'size', 'traverse',
      'answer', 'largest', 'smallest', 'highest', 'lowest', 'longest',
      'shortest', 'count', 'most', 'fewest', 'sum']
  toks = ['_' + w if w in PREDS else w for w in lf.split()]
  return ' '.join(toks)

def data_split(out_data):
  random.shuffle(out_data)
  train_data = out_data[:600]
  test_data = out_data[600:]
  return train_data, test_data

def write(out_basename, out_data):
  out_path = os.path.join(OUT_DIR, out_basename)
  with open(out_path, 'w') as f:
    for x, y in out_data:
      print >> f, '%s\t%s' % (x, y)

def process():
  out_data = []

  in_data = open(IN_FILE).readlines()
  ######  Fill Your Answer  ######
  # Each element in out_data is a (utterance, logical_form) tuple.
  # You will need replace "." with ? in the utterance and use split_logical_form() and reduce_copying() functions on the logical_form
  for inputline in in_data:
    split_list = inputline.split(" ",1);
    # dealing utterance
    utr_list = split_list[0][7:-2].split(",")
    utr_list.pop()
    utr_fin = ""
    for item in utr_list:
      utr_fin += item + " "
    utr_fin += "?" 
    
    # dealing with logic_form
    lf_raw = split_list[1].split(".")[0]
    lf_raw = lf_raw[:-1]
    lf_tem = split_logical_form(lf_raw)
    lf_fin = reduce_copying(lf_tem)

    theTuple = (utr_fin ,lf_fin)
    out_data.append(theTuple)

  train_data, test_data = data_split(out_data)
  train_file = 'geo880_train600.tsv'
  write(train_file, train_data)
  test_file = 'geo880_test280.tsv'
  write(test_file, test_data)

if __name__ == '__main__':
  process()
